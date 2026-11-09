"""Usage-based tariff estimates and conservative cohort spending admission."""
from __future__ import annotations
import datetime as dt
import json
import math
from pathlib import Path
from .transports import estimate_cost


def selected_rates(model: str, catalog: dict, usage: dict | None=None, at: dt.datetime | None=None) -> dict:
    rates=dict(catalog['models'][model])
    if model=='deepseek-v4-pro':
        at=(at or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
        # A verified holiday calendar is not embedded; weekdays retain peak
        # conservatively unless their ordinary UTC off-peak time is clear.
        peak=at.weekday()<5 and (1<=at.hour<4 or 6<=at.hour<10)
        tier='peak' if peak else 'off_peak'
        rates.update(rates['rate_tiers'][tier]);rates['selected_tier']=tier
    context=rates.get('long_context')
    incoming=(usage or {}).get('input_tokens')
    if isinstance(context,dict) and isinstance(incoming,int) and incoming>context['input_tokens_greater_than']:
        rates.update(context['rates']);rates['selected_tier']='long_context'
    return rates


def usage_cost(model: str, usage: dict, catalog: dict, at: dt.datetime | None=None) -> dict:
    rates=selected_rates(model,catalog,usage,at)
    creation=(usage.get('raw') or {}).get('cache_creation',{})
    ttl_delta=0.0
    if isinstance(creation,dict) and any(creation.values()):
        five=creation.get('ephemeral_5m_input_tokens');hour=creation.get('ephemeral_1h_input_tokens')
        valid=lambda x:isinstance(x,int) and not isinstance(x,bool) and x>=0
        if not valid(five) or not valid(hour) or five+hour!=usage.get('cache_write_input_tokens') or 'cache_write_rates' not in rates:
            return {'estimated_usd':None,'quality':'unknown','note':'Cache TTL usage buckets do not match aggregate','rates':rates}
        ttl_delta=hour*(rates['cache_write_rates']['1h']-rates['cache_write_input_per_million'])/1e6
    details=(usage.get('raw') or {}).get('input_tokens_details',{})
    if isinstance(details,dict) and any(('12h' in str(k) or '1h' in str(k)) and bool(v) for k,v in details.items()):
        return {'estimated_usd':None,'quality':'unknown','note':'Nondefault OpenAI cache TTL requires independently pinned rate details','rates':rates}
    amount,note=estimate_cost(usage,rates)
    if amount is not None:
        return {'estimated_usd':amount+ttl_delta,'quality':'reported-usage tariff estimate','note':note,'rates':rates}
    incoming,outgoing=usage.get('input_tokens'),usage.get('output_tokens')
    numeric=lambda x:isinstance(x,int) and not isinstance(x,bool) and x>=0
    if numeric(incoming) and numeric(outgoing):
        # Missing cache splits are not zero cache use. Bill all input at the
        # highest applicable category as an explicitly conservative bound.
        input_rate=max(float(rates[k]) for k in ('input_per_million','cached_input_per_million','cache_write_input_per_million'))
        input_rate=max([input_rate]+[float(v) for v in rates.get('cache_write_rates',{}).values()])
        upper=(incoming*input_rate+outgoing*float(rates['output_per_million']))/1e6
        return {'estimated_usd':upper,'quality':'upper bound; missing cache usage split','note':note,'rates':rates}
    return {'estimated_usd':None,'quality':'unknown','note':note,'rates':rates}


class CostLedger:
    """Known estimates plus unresolved request reserves; neither is an invoice."""
    def __init__(self,path:Path,total_usd:float=1000,gcp_reserve_usd:float=400,contingency_usd:float=100):
        for x in (total_usd,gcp_reserve_usd,contingency_usd):
            if isinstance(x,bool) or not math.isfinite(x) or x<0:raise ValueError('Invalid spending bound')
        if total_usd<=gcp_reserve_usd+contingency_usd:raise ValueError('Budget leaves no API allocation')
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():
            self.data=json.loads(self.path.read_text())
            if (self.data['total_limit_usd'],self.data['gcp_reserve_usd'],self.data['contingency_usd'])!=(total_usd,gcp_reserve_usd,contingency_usd):
                raise ValueError('Existing ledger has different budget bounds')
        else:
            self.data={'schema_version':1,'total_limit_usd':total_usd,'gcp_reserve_usd':gcp_reserve_usd,'contingency_usd':contingency_usd,'requests':[],'invoice_verified':False}
            self.save()

    def save(self):
        temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps(self.data,indent=2,allow_nan=False)+'\n');temp.chmod(0o600);temp.replace(self.path)

    def api_committed(self):
        return sum(r['estimated_usd'] if r.get('estimated_usd') is not None else r['reserved_usd'] for r in self.data['requests'])

    def reserve(self,model:str,phase:str,maximum_usd:float=100)->int:
        if maximum_usd<=0 or not math.isfinite(maximum_usd):raise ValueError('Invalid request reserve')
        if self.api_committed()+maximum_usd+self.data['gcp_reserve_usd']+self.data['contingency_usd']>self.data['total_limit_usd']:
            raise RuntimeError('Cohort spending bound exhausted; no further request authorized')
        i=len(self.data['requests'])
        self.data['requests'].append({'id':i,'model':model,'phase':phase,'reserved_usd':maximum_usd,'estimated_usd':None,'status':'reserved','at':dt.datetime.now(dt.timezone.utc).isoformat()})
        self.save();return i

    def settle(self,index:int,result:dict,catalog:dict):
        row=self.data['requests'][index]
        cost=usage_cost(row['model'],result.get('usage',{}),catalog,dt.datetime.fromisoformat(row['at']))
        amount=cost['estimated_usd']
        if amount is not None and (not math.isfinite(amount) or amount<0):raise ValueError('Invalid usage cost')
        row.update(status='completed',estimated_usd=amount,cost_quality=cost['quality'],request_id=result.get('request_id'),usage=result.get('usage'),model_reported=result.get('model_reported'),service_tier=result.get('service_tier'),speed=result.get('speed'),wall_seconds=result.get('wall_seconds'))
        row['tariff']={k:cost['rates'].get(k) for k in ['input_per_million','output_per_million','cached_input_per_million','cache_write_input_per_million','cache_write_rates','rate_basis','selected_tier'] if k in cost['rates']}
        self.save()
        if amount is not None and amount>row['reserved_usd']:raise RuntimeError('Actual tariff estimate exceeded request reserve; stop cohort')
        return cost

    def fail(self,index:int,error:dict,catalog:dict):
        partial=error.get('partial_result')
        if isinstance(partial,dict) and partial.get('usage'):
            self.settle(index,partial,catalog)
        self.data['requests'][index].update(status='failed',error={k:v for k,v in error.items() if k!='partial_result'})
        if error.get('cost_unknown') is False and error.get('kind') in {'credential_unavailable','credential_loader_required','google_authentication_failed','trial_deadline_exhausted','source_drift','trial_deadline'} and not partial:
            self.data['requests'][index].update(estimated_usd=0.0,cost_quality='No provider generation was made')
        if error.get('kind')=='unexpected_native_tool':
            self.data['requests'][index]['token_only_estimate_usd']=self.data['requests'][index].get('estimated_usd')
            self.data['requests'][index]['estimated_usd']=None
            self.data['requests'][index]['cost_quality']='unknown: unpriced native tool output'
        self.save()

    def summary(self):
        known=sum(r['estimated_usd'] or 0 for r in self.data['requests'])
        unresolved=sum(r['reserved_usd'] for r in self.data['requests'] if r['estimated_usd'] is None)
        return {'known_api_tariff_estimate_usd':known,'unresolved_request_reserves_usd':unresolved,'gcp_reserve_usd':self.data['gcp_reserve_usd'],'contingency_usd':self.data['contingency_usd'],'total_limit_usd':self.data['total_limit_usd'],'invoice_verified':False}
