'use strict';
const $ = id => document.getElementById(id);
let sessionId = null, busy = false;
function notice(msg){let n=$('notice'); if(!n){console.error(msg);return;} n.textContent=msg||'';n.hidden=!msg;}
async function api(path, options={}){
  let response;
  try {response=await fetch(path,{...options,headers:{...(options.body?{'Content-Type':'application/json'}:{}),...(options.headers||{})}});}
  catch(e){throw Error('サーバーに接続できません。');}
  const data=await response.json().catch(()=>({}));
  if(!response.ok)throw Error(data.detail||`HTTP ${response.status}`);
  return data;
}
function disable(flag){busy=flag;document.querySelectorAll('button').forEach(b=>b.disabled=flag);}
function render(state){
  ['pairView','ratingView','resultsView'].forEach(id=>$(id).hidden=true);
  if(state.phase==='pair'){
    $('progressLabel').textContent='STEP 1 / 2';$('countLabel').textContent=`比較 ${state.pair_completed+1} / ${state.pair_total}`;
    $('leftImg').src=state.pair.left.url;$('rightImg').src=state.pair.right.url;
    $('leftName').textContent=state.pair.left.name;$('rightName').textContent=state.pair.right.name;
    $('pairView').hidden=false;
  }else if(state.phase==='rating'){
    $('progressLabel').textContent='STEP 2 / 2';$('countLabel').textContent=`5段階評価 ${state.rating_completed+1} / ${state.rating_total}`;
    $('candidateImg').src=state.candidate.url;$('candidateName').textContent=state.candidate.name;
    $('ratingView').hidden=false;
  }else{
    $('progressLabel').textContent='COMPLETE';$('countLabel').textContent=`比較${state.pair_total}問 ＋ 5段階評価${state.rating_total}問`;
    showResults();
  }
}
async function start(){if(busy)return;disable(true);notice('');try{
  const s=await api('/api/sessions',{method:'POST'});sessionId=s.session_id;
  $('welcome').hidden=true;$('workspace').hidden=false;render(s);
}catch(e){notice(e.message);}finally{disable(false);}}
async function submit(path,payload){if(busy||!sessionId)return;disable(true);notice('');try{
 const s=await api(`/api/sessions/${sessionId}/${path}`,{method:'POST',body:JSON.stringify(payload)});render(s);
}catch(e){notice(e.message);}finally{disable(false);}}
async function showResults(){try{
 const d=await api(`/api/sessions/${sessionId}/recommendations`);const grid=$('resultsGrid');grid.replaceChildren();
 d.items.forEach((item,i)=>{const card=document.createElement('article');card.className='result';const img=document.createElement('img');img.src=item.url;img.alt=item.name;img.loading='lazy';const rank=document.createElement('div');rank.className='rank';rank.textContent=`#${i+1} · Score ${item.score.toFixed(3)}`;const name=document.createElement('p');name.textContent=item.name;card.append(img,rank,name);grid.append(card);});$('resultsView').hidden=false;
}catch(e){notice(e.message);}}
function reset(){sessionId=null;$('welcome').hidden=false;$('workspace').hidden=true;notice('');}
async function init(){try{const d=await api('/api/health');$('datasetBadge').textContent=`${d.images} 枚 · ${d.dimensions}次元`;if(d.missing_images)notice(`${d.missing_images}枚の画像が見つかりません。`);}catch(e){$('datasetBadge').textContent='データ未準備';notice(e.message);}}
// Detect mixed HTML/JavaScript versions instead of crashing on a null element.
const requiredIds=['notice','datasetBadge','welcome','workspace','pairView','ratingView','resultsView','progressLabel','countLabel','leftImg','rightImg','leftName','rightName','candidateImg','candidateName','resultsGrid','startBtn','resetBtn','exportBtn'];
const missingIds=requiredIds.filter(id=>!$(id));
if(missingIds.length){
  const message='画面ファイルのバージョンが一致しません。再デプロイしてブラウザを強制再読み込みしてください。不足ID: '+missingIds.join(', ');
  notice(message);
  throw new Error(message);
}
$('startBtn').addEventListener('click',start);
document.querySelectorAll('[data-choice]').forEach(b=>b.addEventListener('click',()=>submit('votes',{choice:b.dataset.choice})));
document.querySelectorAll('[data-rating]').forEach(b=>b.addEventListener('click',()=>submit('ratings',{rating:Number(b.dataset.rating)})));
$('exportBtn').addEventListener('click',()=>{if(sessionId)location.href=`/api/sessions/${sessionId}/history.csv`;});
$('resetBtn').addEventListener('click',()=>{if(confirm('回答を破棄して最初からやり直しますか？'))reset();});
init();
