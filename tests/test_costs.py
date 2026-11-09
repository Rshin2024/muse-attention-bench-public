import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from muse_attention_bench.costs import CostLedger,selected_rates,usage_cost
from muse_attention_bench.protocol import ROOT

class CostTests(unittest.TestCase):
    def setUp(self):self.catalog=json.loads((ROOT/'configs/pricing.example.json').read_text())
    def test_weekend_deepseek_rate_and_long_openai(self):
        at=dt.datetime(2026,11,8,2,tzinfo=dt.timezone.utc)
        self.assertEqual(selected_rates('deepseek-v4-pro',self.catalog,at=at)['selected_tier'],'off_peak')
        self.assertEqual(selected_rates('gpt-6-astra',self.catalog,{'input_tokens':300000})['input_per_million'],20)
    def test_missing_cache_counters_are_upper_bound_not_zero(self):
        cost=usage_cost('gpt-6-astra',{'input_tokens':1000,'output_tokens':100,'uncached_input_tokens':None,'cached_input_tokens':None,'cache_write_input_tokens':None},self.catalog)
        self.assertGreater(cost['estimated_usd'],0)
        self.assertIn('upper bound',cost['quality'])
    def test_missing_total_usage_stays_unknown(self):
        self.assertIsNone(usage_cost('gpt-6-astra',{},self.catalog)['estimated_usd'])
    def test_anthropic_hour_cache_write_uses_hour_tariff(self):
        usage={'input_tokens':1000000,'output_tokens':0,'uncached_input_tokens':0,'cached_input_tokens':0,'cache_write_input_tokens':1000000,'raw':{'cache_creation':{'ephemeral_5m_input_tokens':0,'ephemeral_1h_input_tokens':1000000}}}
        self.assertEqual(usage_cost('claude-fable-5-1',usage,self.catalog)['estimated_usd'],20)
    def test_budget_reserves_unknown_calls_and_persists(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ledger.json';ledger=CostLedger(path,300,100,50)
            ledger.reserve('gpt-6-astra','test',100)
            with self.assertRaises(RuntimeError):ledger.reserve('gpt-6-astra','test',100)
            self.assertEqual(CostLedger(path,300,100,50).summary()['unresolved_request_reserves_usd'],100)
    def test_known_usage_releases_unused_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=CostLedger(Path(d)/'ledger.json',300,100,50);i=ledger.reserve('gpt-6-astra','test',100)
            ledger.settle(i,{'usage':{'input_tokens':100,'output_tokens':10,'uncached_input_tokens':100,'cached_input_tokens':0,'cache_write_input_tokens':0}},self.catalog)
            self.assertLess(ledger.api_committed(),1)
            self.assertEqual(ledger.summary()['unresolved_request_reserves_usd'],0)
    def test_auth_failure_before_request_is_not_unknown_spending(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=CostLedger(Path(d)/'ledger.json');i=ledger.reserve('gpt-6-astra','test',100)
            ledger.fail(i,{'kind':'credential_unavailable','cost_unknown':False},self.catalog)
            self.assertEqual(ledger.summary()['known_api_tariff_estimate_usd'],0)
            self.assertEqual(ledger.summary()['unresolved_request_reserves_usd'],0)

if __name__=='__main__':unittest.main()
