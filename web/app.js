"use strict";
const $=id=>document.getElementById(id);
let sessionId=null, busy=false, pair=null;
function notice(message){const n=$("notice");n.textContent=message||"";n.hidden=!message; if(message)n.scrollIntoView({behavior:"smooth",block:"nearest"});}
async function api(path,options={}){let response;try{response=await fetch(path,{...options,headers:{...(options.body?{"Content-Type":"application/json"}:{}),...(options.headers||{})}});}catch(e){throw new Error("サーバーに接続できません。通信状態を確認してください。");}let data=await response.json().catch(()=>({}));if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:`HTTP ${response.status}`);return data;}
function setBusy(state){busy=state;document.querySelectorAll('button').forEach(b=>b.disabled=state);}
function showPair(p,count){pair=p;$("roundLabel").textContent=`Round ${p.round}`;$("countLabel").textContent=`${count} 回回答`;$("leftImg").src=p.left.url;$("rightImg").src=p.right.url;$("leftName").textContent=p.left.name;$("rightName").textContent=p.right.name;$("compareView").hidden=false;$("resultsView").hidden=true;}
async function start(){if(busy)return;setBusy(true);notice("");try{const data=await api('/api/sessions',{method:'POST'});sessionId=data.session_id;$("welcome").hidden=true;$("workspace").hidden=false;showPair(data.pair,0);}catch(e){notice(e.message);}finally{setBusy(false);}}
async function vote(choice){if(busy||!sessionId||!pair)return;setBusy(true);notice("");try{const d=await api(`/api/sessions/${sessionId}/votes`,{method:'POST',body:JSON.stringify({choice})});showPair(d.pair,d.count);}catch(e){notice(e.message);}finally{setBusy(false);}}
async function results(){if(busy||!sessionId)return;setBusy(true);notice("");try{const d=await api(`/api/sessions/${sessionId}/recommendations?limit=12`);$("trainedNote").textContent=d.trained?`${d.count} 回の回答から推定した好みの近い順です。`:'まだ評価がありません。まず比較に回答してください。';const grid=$("resultsGrid");grid.replaceChildren();d.items.forEach((item,i)=>{const card=document.createElement('article');card.className='result';const img=document.createElement('img');img.src=item.url;img.alt=item.name;img.loading='lazy';const rank=document.createElement('div');rank.className='rank';rank.textContent=`#${i+1}${d.trained?' · 類似度 '+item.score.toFixed(3):''}`;const label=document.createElement('p');label.textContent=item.name;card.append(img,rank,label);grid.append(card);});$("compareView").hidden=true;$("resultsView").hidden=false;}catch(e){notice(e.message);}finally{setBusy(false);}}
function reset(){sessionId=null;pair=null;$("workspace").hidden=true;$("welcome").hidden=false;notice("");}
async function init(){try{const d=await api('/api/health');$("datasetBadge").textContent=`${d.images} 枚 · ${d.dimensions}次元`;if(d.missing_images)notice(`画像が ${d.missing_images} 枚見つかりません。Eyemake4 フォルダを確認してください。`);}catch(e){$("datasetBadge").textContent='データ未準備';notice(`バックエンドのデータ読み込みエラー: ${e.message}\nEyemake4/clip_embeddings.npz と画像を確認してください。`);}}
$("startBtn").addEventListener('click',start);
document.querySelectorAll('[data-vote]').forEach(b=>b.addEventListener('click',()=>vote(b.dataset.vote)));
$("resultsBtn").addEventListener('click',results);
$("backBtn").addEventListener('click',()=>{$("compareView").hidden=false;$("resultsView").hidden=true;});
$("resetBtn").addEventListener('click',()=>{if(confirm('新しいセッションを開始しますか？現在の学習履歴は引き継がれません。'))reset();});
$("exportBtn").addEventListener('click',()=>{if(sessionId)window.location.href=`/api/sessions/${sessionId}/history.csv`;});
init();