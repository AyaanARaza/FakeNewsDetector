import * as ort from './ort.wasm.bundle.min.mjs';

document.getElementById('year').textContent = new Date().getFullYear();

(async () => {
  const btn = document.querySelector('.scan-button');
  const card = document.getElementById('resultCard');
  const nbEl = document.getElementById('nbRes');
  const svmEl = document.getElementById('svmRes');
  const xgbEl = document.getElementById('xgbRes');
  const majEl = document.getElementById('majRes');

  const vectorizerUrl = chrome.runtime.getURL('vectorizer.json');
  const resp = await fetch(vectorizerUrl);
  const rawText = await resp.text();

  let data;
  try {
    data = JSON.parse(rawText);
  } catch (e) {
    console.error('❌ Failed to parse vectorizer.json:', e);
    throw e;
  }

  const vocabKey = data.vocabulary ? 'vocabulary'
    : data.vocabulary_ ? 'vocabulary_'
      : data.vocab ? 'vocab'
        : null;

  if (!vocabKey) {
    console.error('❌ No known vocab key found; keys were:', Object.keys(data));
    return;
  }

  const vocab = data;

  const idf = data.idf || null;

  const [nbSession, svmSession, xgbSession] = await Promise.all([
    ort.InferenceSession.create(chrome.runtime.getURL('models/nb.onnx')),
    ort.InferenceSession.create(chrome.runtime.getURL('models/svm.onnx')),
    ort.InferenceSession.create(chrome.runtime.getURL('models/xgboost.onnx'))
  ]);

  function textToVector(txt) {
    const tokens = txt.match(/\b\w+\b/g) || [];
    const vec = new Float32Array(Object.keys(vocab).length);
    for (let t of tokens) {
      const idx = vocab[t];
      if (idx !== undefined) vec[idx]++;
    }
    if (idf) {
      for (let i = 0; i < vec.length; i++) {
        vec[i] *= idf[i];
      }
    }
    return vec;
  }

  async function runModels(text) {
    const rawVec = textToVector(text);
    const featureCount = nbSession.inputMetadata[0].shape[1];
    const vec = new Float32Array(featureCount);
    vec.set(rawVec.subarray(0, featureCount));
    const tensor = new ort.Tensor('float32', vec, [1, featureCount]);

    const nbFeeds = { [nbSession.inputNames[0]]: tensor };
    const svmFeeds = { [svmSession.inputNames[0]]: tensor };
    const xgbFeeds = { [xgbSession.inputNames[0]]: tensor };

    const [nbOut, svmOut, xgbOut] = await Promise.all([
      nbSession.run(nbFeeds),
      svmSession.run(svmFeeds),
      xgbSession.run(xgbFeeds),
    ]);


    const nb = Number(nbOut.label.cpuData[0])
    const svm = Number(svmOut.label.cpuData[0]);
    const xgb = Number(xgbOut.label.cpuData[0]);

    return { nb, svm, xgb };
  }





  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.textContent = 'Scanning…';

    const [tab] = await chrome.tabs.query({
      active: true,
      lastFocusedWindow: true
    });

    if (!tab || !tab.id) {
      console.error('❌ No active tab found');
      btn.textContent = 'Error';
      btn.disabled = false;
      return;
    }

    chrome.tabs.sendMessage(
      tab.id,
      { action: 'getPageText' },
      async response => {
        if (!response || !response.body) {
          console.error('❌ No response from content script');
          btn.textContent = 'Error';
          btn.disabled = false;
          return;
        }

        const text = (response.title + ' ' + response.body).toLowerCase();

        try {
          const { nb, svm, xgb } = await runModels(text);
          nbEl.textContent = `Naive Bayes: ${nb == 1 ? 'FAKE' : 'REAL'}`;
          svmEl.textContent = `SVM: ${svm == 1 ? 'FAKE' : 'REAL'}`;
          xgbEl.textContent = `XGBoost: ${xgb == 1 ? 'FAKE' : 'REAL'}`;

          const vote = [nb, svm, xgb].filter(p => p > 0.5).length >= 2 ? 'FAKE' : 'REAL';
          majEl.textContent = `Majority: ${vote}`;

          card.classList.add('show');
        } catch (e) {
          console.error('❌ Error running models:', e);
          btn.textContent = 'Error';
        }

        btn.textContent = 'Scan Again';
        btn.disabled = false;
      }
    );
  });
})();


