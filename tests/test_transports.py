import copy
import io
import json
import subprocess
import unittest
from urllib.error import HTTPError

from muse_attention_bench.providers import load_configuration
from muse_attention_bench.transports import (
    TransportError, build_request, estimate_cost, generate, normalize_response,
    normalize_usage,
    _NoRedirect,
)


class Response:
    def __init__(self, body, headers=None):
        self.body=json.dumps(body).encode();self.headers=headers or {}
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self,size):return self.body[:size]


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.providers={p['id']:p for p in load_configuration()['providers']}
        self.messages=[{'role':'system','content':'Return exactly one JSON action.'},
                       {'role':'user','content':'Read TASK.md.'}]
        self.responses={'id':'resp-fixture','model':'gpt-6-astra','status':'completed',
            'output':[{'type':'reasoning','encrypted_content':'opaque-private-state','summary':[]},
                      {'type':'message','role':'assistant','content':[{'type':'output_text','text':'{"done":'},{'type':'output_text','text':'false}'}]}],
            'usage':{'input_tokens':100,'input_tokens_details':{'cached_tokens':20,'cache_write_tokens':0},
                     'output_tokens':40,'output_tokens_details':{'reasoning_tokens':25},'total_tokens':140}}

    def plan(self,provider,**kwargs):
        p=self.providers[provider]
        return build_request(p,p['model'],self.messages,
            environment={'GOOGLE_CLOUD_PROJECT':'test-project','GOOGLE_CLOUD_LOCATION':'global'},**kwargs)

    def test_all_seven_exact_models_have_native_request_plans(self):
        models=[]
        for provider,entry in self.providers.items():
            for model in entry['configured_models']:
                p=build_request(entry,model,self.messages,environment={'GOOGLE_CLOUD_PROJECT':'test-project'})
                self.assertEqual(p['provider'],provider)
                self.assertTrue(p['url'].startswith('https://'))
                models.append(model)
        self.assertEqual(len(models),7)

    def test_responses_are_stateless_no_tools_high_reasoning(self):
        for provider in ['openai','deepseek']:
            p=self.plan(provider);b=p['body']
            self.assertEqual(b['reasoning']['effort'],'high')
            self.assertFalse(b['store']);self.assertFalse(b['stream'])
            self.assertEqual(b['tools'],[]);self.assertEqual(b['tool_choice'],'none')
            self.assertEqual(b['max_output_tokens'],32768)
            self.assertEqual(b['text']['format']['type'],'json_object')
        self.assertIn('include',self.plan('openai')['body'])
        self.assertEqual(self.plan('openai')['body']['service_tier'],'default')
        self.assertNotIn('include',self.plan('deepseek')['body'])

    def test_structured_schema_uses_only_native_documented_fields(self):
        schema={'type':'object','properties':{},'additionalProperties':False}
        o=self.plan('openai',action_schema=schema)['body']['text']['format']
        d=self.plan('deepseek',action_schema=schema)['body']['text']['format']
        self.assertTrue(o['strict'])
        self.assertNotIn('strict',d)

    def test_fable_adaptive_no_forced_tools_or_sampling(self):
        p=build_request(self.providers['anthropic'],'claude-fable-5-1',self.messages)
        self.assertEqual(p['body']['thinking'],{'type':'adaptive'})
        self.assertEqual(p['body']['output_config']['effort'],'high')
        self.assertNotIn('tool_choice',p['body'])
        self.assertNotIn('temperature',p['body'])
        self.assertEqual(p['body']['system'],self.messages[0]['content'])

    def test_google_maps_system_and_high_json(self):
        p=self.plan('gemini');b=p['body']
        self.assertNotIn('model',b)
        self.assertIn('gemini-4-argon:generateContent',p['url'])
        self.assertEqual(b['contents'][0]['role'],'user')
        self.assertEqual(b['systemInstruction']['parts'][0]['text'],self.messages[0]['content'])
        self.assertEqual(b['generationConfig']['thinkingConfig']['thinkingLevel'],'HIGH')
        self.assertEqual(b['generationConfig']['responseMimeType'],'application/json')
        self.assertNotIn('temperature',b['generationConfig'])

    def test_google_does_not_silently_switch_an_unimplemented_auth_flow(self):
        p=copy.deepcopy(self.providers['gemini']);p['authentication_selection']='google_adc'
        with self.assertRaises(ValueError):
            build_request(p,p['model'],self.messages,environment={'GOOGLE_CLOUD_PROJECT':'test-project'})

    def test_glm_native_json_and_thinking(self):
        p=self.plan('glm');b=p['body']
        self.assertEqual(b['thinking'],{'type':'enabled'})
        self.assertEqual(b['reasoning_effort'],'high')
        self.assertEqual(b['response_format'],{'type':'json_object'})
        self.assertEqual(b['messages'],self.messages)

    def test_native_roles_do_not_downgrade_developer_instructions(self):
        m=copy.deepcopy(self.messages);m[0]['role']='developer'
        for provider in ['deepseek','glm']:
            p=build_request(self.providers[provider],self.providers[provider]['model'],m)
            self.assertEqual(p['history'][0]['role'],'system')

    def test_json_mode_checks_shared_prompt_before_paid_request(self):
        m=[{'role':'user','content':'Hello'}]
        with self.assertRaises(ValueError):build_request(self.providers['openai'],'gpt-6-astra',m)

    def test_proxy_or_unconfigured_model_rejected(self):
        p=copy.deepcopy(self.providers['openai']);p['base_url']='https://not-openai.example/v1'
        with self.assertRaises(ValueError):build_request(p,p['model'],self.messages)
        with self.assertRaises(ValueError):build_request(self.providers['openai'],'latest',self.messages)

    def test_redirects_cannot_forward_credentials(self):
        self.assertIsNone(_NoRedirect().redirect_request(None,None,302,'redirect',{},'https://attacker.example'))

    def test_responses_output_and_usage_do_not_expose_reasoning(self):
        plan=build_request(self.providers['openai'],'gpt-6-astra',self.messages)
        r=normalize_response(self.providers['openai'],'gpt-6-astra',plan,self.responses)
        self.assertEqual(r['output_text'],'{"done":false}')
        self.assertNotIn('opaque-private-state',r['output_text'])
        self.assertEqual(r['usage']['output_tokens'],40)
        self.assertEqual(r['usage']['reasoning_tokens'],25)
        self.assertEqual(r['usage']['uncached_input_tokens'],80)
        self.assertIsNone(r['estimated_cost_usd'])

    def test_request_and_response_ids_remain_distinct(self):
        p=build_request(self.providers['openai'],'gpt-6-astra',self.messages)
        r=normalize_response(self.providers['openai'],'gpt-6-astra',p,self.responses,response_headers={'X-Request-Id':'req-fixture'})
        self.assertEqual(r['request_id'],'req-fixture')
        self.assertEqual(r['response_id'],'resp-fixture')

    def test_native_responses_history_is_replayed_once(self):
        entry=self.providers['openai'];model='gpt-6-astra'
        p=build_request(entry,model,self.messages)
        r=normalize_response(entry,model,p,self.responses)
        history=self.messages+[{'role':'assistant','content':r['output_text']},
                               {'role':'user','content':'Command completed.'}]
        p2=build_request(entry,model,history,native_state=r['native_state'])
        self.assertEqual(len(p2['history']),len(p['history'])+len(self.responses['output'])+1)
        self.assertEqual(sum(x.get('type')=='reasoning' for x in p2['history']),1)
        self.assertEqual(p2['history'][-1]['content'],'Command completed.')
        edited=copy.deepcopy(history);edited[1]['content']='Changed old context'
        with self.assertRaises(ValueError):build_request(entry,model,edited,native_state=r['native_state'])
        edited_state=copy.deepcopy(r['native_state']);edited_state['history'][2]['encrypted_content']='edited'
        with self.assertRaises(ValueError):build_request(entry,model,history,native_state=edited_state)

    def test_anthropic_preserves_private_thinking_blocks(self):
        entry=self.providers['anthropic'];p=self.plan('anthropic')
        response={'id':'msg-test','model':entry['model'],'stop_reason':'end_turn',
            'content':[{'type':'thinking','thinking':'private reasoning','signature':'opaque-signature'},
                       {'type':'text','text':'{"done":false}'}],
            'usage':{'input_tokens':100,'cache_read_input_tokens':20,'cache_creation_input_tokens':10,
                     'output_tokens':40,'output_tokens_details':{'thinking_tokens':25}}}
        r=normalize_response(entry,entry['model'],p,response)
        self.assertNotIn('private reasoning',r['output_text'])
        self.assertEqual(r['usage']['input_tokens'],130)
        m=self.messages+[{'role':'assistant','content':r['output_text']},{'role':'user','content':'ok'}]
        p2=build_request(entry,entry['model'],m,native_state=r['native_state'])
        self.assertEqual(p2['body']['messages'][-2]['content'],response['content'])

    def test_google_preserves_thought_signature_and_counts_thinking(self):
        entry=self.providers['gemini'];p=self.plan('gemini')
        response={'responseId':'google-test','modelVersion':'gemini-4-argon',
            'candidates':[{'content':{'role':'model','parts':[{'thought':True,'text':'private thought'},
                {'text':'{"done":false}','thoughtSignature':'opaque'}]},'finishReason':'STOP'}],
            'usageMetadata':{'promptTokenCount':100,'cachedContentTokenCount':20,'candidatesTokenCount':12,'thoughtsTokenCount':28,'totalTokenCount':140}}
        r=normalize_response(entry,entry['model'],p,response)
        self.assertNotIn('private thought',r['output_text'])
        self.assertEqual(r['usage']['output_tokens'],40)
        self.assertEqual(r['usage']['uncached_input_tokens'],80)
        self.assertEqual(r['native_state']['history'][-1]['parts'][1]['thoughtSignature'],'opaque')

    def test_glm_usage_not_double_counted(self):
        u=normalize_usage('glm',{'usage':{'prompt_tokens':100,'completion_tokens':40,
            'prompt_tokens_details':{'cached_tokens':20},'completion_tokens_details':{'reasoning_tokens':25},'total_tokens':140}})
        self.assertEqual(u['output_tokens'],40)
        self.assertEqual(u['reasoning_tokens'],25)

    def test_explicit_rates_and_missing_rates(self):
        u=normalize_usage('openai',self.responses)
        rates={'input_per_million':10,'cached_input_per_million':1,'output_per_million':50}
        self.assertAlmostEqual(estimate_cost(u,rates)[0],.00282)
        self.assertIsNone(estimate_cost(u,{'input_per_million':10,'output_per_million':50})[0])
        self.assertIsNone(estimate_cost(u,None)[0])

    def test_missing_openai_cache_write_counter_is_not_zero_cost(self):
        r=copy.deepcopy(self.responses);r['usage']['input_tokens_details'].pop('cache_write_tokens')
        u=normalize_usage('openai',r)
        self.assertIsNone(u['cache_write_input_tokens'])
        self.assertIsNone(estimate_cost(u,{'input_per_million':10,'output_per_million':50,'cached_input_per_million':1})[0])

    def test_generation_auth_and_timeout_are_bounded(self):
        seen=[]
        def opener(req,timeout):
            seen.append((req,timeout));return Response(self.responses)
        r=generate(self.providers['openai'],'gpt-6-astra',self.messages,5,
                   lambda p:'fixture-key',opener=opener,timeout_cap=120)
        self.assertEqual(len(seen),1)
        self.assertLessEqual(seen[0][1],5)
        self.assertEqual(seen[0][0].get_header('Authorization'),'Bearer fixture-key')
        self.assertEqual(r['request_id'],'resp-fixture')
        self.assertFalse(r['automatically_retried'])

    def test_google_auth_uses_fixed_trusted_command_only(self):
        commands=[]
        def runner(args,**kwargs):
            commands.append(args);return subprocess.CompletedProcess(args,0,'fixture-google-token\n','')
        response={'candidates':[{'content':{'role':'model','parts':[{'text':'{}'}]},'finishReason':'STOP'}],
                  'usageMetadata':{'promptTokenCount':2,'candidatesTokenCount':2}}
        r=generate(self.providers['gemini'],'gemini-4-argon',self.messages,5,
            environment={'GOOGLE_CLOUD_PROJECT':'fixture-project'},
            token_command_runner=runner,opener=lambda req,timeout:Response(response))
        self.assertEqual(commands,[['gcloud','auth','print-access-token']])
        self.assertEqual(r['output_text'],'{}')

    def test_http_errors_are_not_retried_or_echoed(self):
        calls=[]
        def opener(req,timeout):
            calls.append(req)
            raise HTTPError(req.full_url,429,'provider body contains fixture-key',{},io.BytesIO(b'fixture-key prompt'))
        with self.assertRaises(TransportError) as caught:
            generate(self.providers['openai'],'gpt-6-astra',self.messages,5,lambda p:'fixture-key',opener=opener)
        self.assertEqual(len(calls),1)
        encoded=json.dumps(caught.exception.as_dict())+str(caught.exception)
        self.assertNotIn('fixture-key',encoded)
        self.assertNotIn('prompt',encoded)
        self.assertTrue(caught.exception.retryable)

    def test_expired_trial_never_loads_credentials_or_calls_network(self):
        def fail(*args,**kwargs):raise AssertionError('must not execute')
        with self.assertRaises(TransportError):
            generate(self.providers['openai'],'gpt-6-astra',self.messages,0,fail,opener=fail)

    def test_native_tool_response_keeps_usage_but_does_not_execute(self):
        p=build_request(self.providers['openai'],'gpt-6-astra',self.messages)
        r=copy.deepcopy(self.responses);r['output'].append({'type':'function_call','name':'shell','arguments':'...','call_id':'x'})
        with self.assertRaises(TransportError) as caught:
            normalize_response(self.providers['openai'],'gpt-6-astra',p,r)
        self.assertEqual(caught.exception.kind,'unexpected_native_tool')
        self.assertEqual(caught.exception.partial_result['usage']['output_tokens'],40)
        self.assertNotIn('native_state',caught.exception.partial_result)

    def test_malformed_paid_response_preserves_usage(self):
        p=build_request(self.providers['openai'],'gpt-6-astra',self.messages)
        r=copy.deepcopy(self.responses);r['output']=[None]
        with self.assertRaises(TransportError) as caught:
            normalize_response(self.providers['openai'],'gpt-6-astra',p,r)
        self.assertEqual(caught.exception.kind,'malformed_response')
        self.assertEqual(caught.exception.partial_result['usage']['output_tokens'],40)


if __name__=='__main__':unittest.main()
