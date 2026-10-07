// Exercise the shipped UI functions with controlled responses and a small DOM.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const copy = value => JSON.parse(JSON.stringify(value));
const config = {
  dataset: {gold:'upload:gold.csv',word_column:'word',score_column:'score',lowercase:true,pos_filter:{enabled:false,tags:['Noun']}},
  embeddings: {mode:'single',active_space:'arb_space',spaces:[{id:'arb_space',label:'Arabic',path:'embedding:arb_space',kind:'ft_bin'}]},
  runtime: {seed:13,n_jobs:1},prediction:{target:'upload:target.txt'},reports:{level:'full'},
};
const candidates = [
  {id:'arb_space',label:'Arabic',path:'embedding:arb_space',kind:'ft_bin'},
  {id:'de_space',label:'German',path:'embedding:de_space',kind:'vec'},
];

function setup({hosted = false} = {}) {
  const nodes = new Map(), timers = new Map(), requests = [];
  let timerId = 0;
  class Element {
    constructor() {
      this.value = ''; this.textContent = ''; this.children = []; this.handlers = {};
      this.customValidity = ''; this.focused = false;
      this.classList = {add(){},remove(){},toggle(){return false;}};
    }
    appendChild(child) {this.children.push(child);}
    set innerHTML(value) {this.children = []; this.html = value;}
    get innerHTML() {return this.html || '';}
    get options() {return this.children;}
    addEventListener(type, fn) {this.handlers[type] = fn;}
    querySelector() {return new Element();}
    closest() {return null;}
    setCustomValidity(value) {this.customValidity = value;}
    checkValidity() {return !this.customValidity;}
    reportValidity() {return this.checkValidity();}
    focus() {this.focused = true;}
    click() {return this.handlers.click?.({target:this});}
  }
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, new Element());
    return nodes.get(id);
  };
  const env = {
    confirm: true, confirmCount: 0,
    respond: async url => url.includes('/config/default') ? {raw_config:copy(config)}
      : url.includes('/fs/embeddings') ? {candidates:copy(candidates)}
      : url.includes('/jobs/active') ? {active_job:null} : {latest_job:null},
  };
  const context = {
    document: {body:{dataset:{hosted:String(hosted)}},getElementById:node,querySelector:()=>null,querySelectorAll:()=>[],
      createElement:()=>new Element(),addEventListener(){}},
    window: {confirm(){env.confirmCount++;return env.confirm;},addEventListener(){},location:{}},
    Element, Headers, URLSearchParams, FormData, Date, AbortController,
    fetch: async (...args) => {
      requests.push(args);
      const payload = await env.respond(...args);
      return {ok:true,status:200,json:async()=>payload,text:async()=>JSON.stringify(payload)};
    },
    setTimeout(fn,ms){timers.set(++timerId,{fn,ms});return timerId;},
    clearTimeout(id){timers.delete(id);},setInterval(){return 1;},clearInterval(){},console,
  };
  assert.ok(source.includes('  boot();'));
  vm.createContext(context);
  vm.runInContext(source.replace('  boot();', `globalThis.ui={state,formatCsvCellValue,setRawConfig,buildConfigFromGuided,
    handleNewConfig,handleLoadConfig,handleUploadConfig,handleSaveConfig,pollJob,recoverActiveJob,
    previewSelectedArtifact,refreshLatestResults,bindEvents};`), context);
  const ui = context.ui;
  ui.setRawConfig(copy(config),'test',true,'session-config:saved.json');
  // Candidate metadata is normally loaded asynchronously when a config opens.
  ui.state.embeddingCandidates = copy(candidates);
  return {ui,node,env,timers,requests,context};
}

test('model selection updates all metadata before export or running', () => {
  const {ui,node} = setup();
  node('emb_single_path').value = 'embedding:de_space';
  assert.deepEqual(copy(ui.buildConfigFromGuided().embeddings), {
    mode:'single',active_space:'de_space',spaces:[{id:'de_space',label:'German',path:'embedding:de_space',kind:'vec'}],
  });
});

test('word and neighbor text remains literal, only scores are formatted', () => {
  const {ui} = setup();
  for (const word of ['001','12345','1e3','A|B','001 | Haus',"O’Neill",'東京']) {
    for (const column of ['word','neighbors','neighbors__de_space']) {
      assert.equal(ui.formatCsvCellValue(word,column),word);
    }
  }
  assert.equal(ui.formatCsvCellValue('5.12345','pred'),'5.12');
  assert.equal(ui.formatCsvCellValue('5.12345 | 4.12345','neighbor_gold_scores'),'5.12 | 4.12');
});

test('canceling New, Load and Upload preserves the edited config', async () => {
  const {ui,node,env} = setup();
  ui.state.dirty = true; env.confirm = false;
  node('dataset_word_column').value = 'unsaved';
  await ui.handleNewConfig(); await ui.handleLoadConfig('another.json'); await ui.handleUploadConfig({});
  assert.equal(env.confirmCount,3);
  assert.equal(node('dataset_word_column').value,'unsaved');
  assert.equal(node('configPath').value,'session-config:saved.json');
});

test('New clears the old save destination and does not claim to be saved', async () => {
  const {ui,node,env} = setup();
  ui.state.dirty = true;
  await ui.handleNewConfig();
  assert.equal(env.confirmCount,1);
  assert.equal(node('configPath').value,'');
  assert.match(node('editorSessionState').textContent,/not saved yet/);
  await assert.rejects(ui.handleSaveConfig(),/path/i);
});

test('invalid numbers stop saving and focus the invalid field', async () => {
  for (const [field,value] of [['prediction_k_min','5.9'],['prediction_cv_folds',''],['prediction_test_size','1'],['runtime_n_jobs','0']]) {
    const {ui,node,requests} = setup();
    node(field).value = value;
    await assert.rejects(ui.handleSaveConfig());
    assert.equal(node(field).focused,true);
    assert.ok(node(field).customValidity);
    assert.ok(!requests.some(([url])=>url.includes('/config/save')));
  }
});

test('hosted configs use supported server settings', () => {
  const {ui} = setup({hosted:true});
  const imported = copy(config);
  imported.runtime.n_jobs = -1; imported.reports.level = 'core';
  imported.dataset.pos_filter.token_pattern = 'custom';
  ui.setRawConfig(imported,'import',false);
  const rebuilt = ui.buildConfigFromGuided();
  assert.equal(rebuilt.runtime.n_jobs,1);
  assert.equal(rebuilt.reports.level,'full');
  assert.equal(rebuilt.dataset.pos_filter.token_pattern,'[,;/| ]+');
});

function prepareResults(ui) {
  ui.state.results.latestJob = {job_id:'test',artifacts:{vocab_predictions:'vocab',summary:'summary'}};
  ui.state.results.artifactKey = 'vocab_predictions';
}
function preview(label) {
  return {kind:'text',offset:0,total_matches:1,has_more:false,artifact_path:label,lines:[{line_number:1,text:label}]};
}

test('Next resets pagination when the search query has changed', async () => {
  const {ui,node,env,requests} = setup();
  prepareResults(ui); ui.bindEvents();
  ui.state.results.query = 'a'; ui.state.results.offset = 50;
  node('resultSearchQuery').value = 'Fußangel';
  env.respond = async () => preview('matching row');
  await node('btnNextPage').click();
  const url = new URL(requests.at(-1)[0],'https://test');
  assert.equal(url.searchParams.get('q'),'Fußangel');
  assert.equal(url.searchParams.get('offset'),'0');
  assert.equal(node('resultPagingInfo').textContent,'1-1 / 1');
});

test('an older search response cannot replace a newer result', async () => {
  const {ui,node,env} = setup();
  prepareResults(ui);
  const pending = [];
  env.respond = () => new Promise(resolve=>pending.push(resolve));
  node('resultSearchQuery').value = 'first'; const first = ui.previewSelectedArtifact(true);
  node('resultSearchQuery').value = 'second'; const second = ui.previewSelectedArtifact(true);
  pending[1](preview('second')); await second;
  pending[0](preview('first')); await first;
  assert.equal(node('resultPath').textContent,'second');
});

test('an older artifact response cannot replace the selected summary', async () => {
  const {ui,node,env} = setup();
  prepareResults(ui);
  const pending = [];
  env.respond = () => new Promise(resolve=>pending.push(resolve));
  const csv = ui.previewSelectedArtifact(true);
  ui.state.results.artifactKey = 'summary'; const summary = ui.previewSelectedArtifact(true);
  pending[1]({test_rmse:1}); await summary;
  pending[0](preview('old csv')); await csv;
  assert.equal(node('resultPath').textContent,'summary');
  assert.equal(node('resultPagingInfo').textContent,'summary');
});

test('a superseded request failure does not replace the new result with an error', async () => {
  const {ui,node,env} = setup();
  prepareResults(ui);
  const pending = [];
  env.respond = () => new Promise((resolve,reject)=>pending.push({resolve,reject}));
  const first = ui.previewSelectedArtifact(true);
  node('resultSearchQuery').value = 'second'; const second = ui.previewSelectedArtifact(true);
  pending[1].resolve(preview('second')); await second;
  pending[0].reject(new Error('Old request failed')); await first;
  assert.equal(node('resultPath').textContent,'second');
});

test('a failed status request retries and recovers completion', async () => {
  const {ui,node,env,timers} = setup();
  env.respond = async () => ({job_id:'test',status:'running',created_at:new Date().toISOString(),logs:[]});
  await ui.pollJob('test');
  assert.equal(node('btnRunPrediction').disabled,true);
  env.respond = async () => {throw new Error('Connection lost');};
  await ui.pollJob('test');
  assert.equal(node('workingBadge').textContent,'RECONNECTING');
  assert.ok([...timers.values()].some(timer=>timer.ms >= 1000 && timer.ms <= 30000));
  env.respond = async url => url.includes('/jobs/') ? {job_id:'test',status:'done',logs:[]} : {latest_job:null};
  await [...timers.values()].at(-1).fn();
  assert.equal(node('workingBadge').textContent,'DONE');
  assert.equal(node('btnRunPrediction').disabled,false);
  assert.equal(ui.state.currentJobId,null);
});

test('reload recovery discovers and polls the session active job', async () => {
  const {ui,env,requests,node} = setup();
  env.respond = async url => url.endsWith('/active')
    ? {active_job:{job_id:'existing',status:'queued',logs:[]}}
    : {job_id:'existing',status:'running',logs:[]};
  await ui.recoverActiveJob();
  assert.ok(requests.some(([url])=>url==='/api/jobs/existing'));
  assert.equal(ui.state.currentJobId,'existing');
  assert.equal(node('btnRunPrediction').disabled,true);
});
