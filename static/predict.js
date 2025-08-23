const VIDEO = document.getElementById('webcam');
const progressBar = document.getElementById("progressbar");
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const CLASS_NAMES = [];

let mobilenet = undefined;
let videoPlaying = false;
let predict = false;
let model;

mobilenet = await loadMobileNetFeatureModel();

async function init() {
  enableCam();

  const res = await fetch(`/model_exists/${window.modelId}`);
  const data = await res.json();

  if (data.exists) {
    console.log("Loading saved model...");
    model = await tf.loadLayersModel(
      `http://127.0.0.1:5000/static/models/${window.modelId}/model.json`
    );
    const namesRes = await fetch(`/models/${window.modelId}/class_names.json`);
    const namesData = await namesRes.json();
    CLASS_NAMES.splice(0, CLASS_NAMES.length, ...namesData);
    createPredictionUI(true);
    predict = true;
    await new Promise(r => setTimeout(r, 500));
    predictLoop();
  }
  else {
    enableCam();
    console.log("No model found.");
    rebuildModel();
  }
}

init();

async function enableCam() {
  if (hasGetUserMedia()) {
    const constraints = {
      video: true,
      width: 640,
      height: 480
    };

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
      VIDEO.srcObject = stream;
      VIDEO.addEventListener('loadeddata', function () {
        videoPlaying = true;
      });
    });
  } else {
    console.warn('getUserMedia() is not supported by your browser');
    alert("You need to enable the camera in order to continue")
  }
}

async function loadMobileNetFeatureModel() {
  const URL =
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';

  mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
  document.getElementById("loading-screen").style.display = 'none';

  tf.tidy(function () {
    let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log(`Mobilenet Output:${answer.shape}`);
  });
  return mobilenet;
}

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

async function predictLoop() {
  if (predict && videoPlaying) {
    let prediction;
    let imageFeatures;


    imageFeatures = tf.tidy(() => {
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
      let resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor,
        [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
        true
      );
      let normalizedTensorFrame = resizedTensorFrame.div(255);
      return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
    });

    prediction = model.predict(imageFeatures.expandDims());
    const predData = await prediction.data();
    if (CLASS_NAMES.length === 2) {
      const class1Confidence = Math.ceil((predData[0]) * 100);
      const class0Confidence = Math.ceil((1 - predData[0]) * 100);
      const bar0 = document.getElementById('progressbar_0');
      const bar1 = document.getElementById('progressbar_1');

      if (bar0 && bar1) {
        bar0.innerHTML = class0Confidence + '%';
        bar1.innerHTML = class1Confidence + '%';
        bar0.style.width = class0Confidence + '%';
        bar1.style.width = class1Confidence + '%';
        bar0.setAttribute("aria-valuenow", class0Confidence);
        bar1.setAttribute("aria-valuenow", class1Confidence);
      }
    }
    else {
      predData.forEach((score, index) => {
        const confidence = Math.ceil(score * 100);
        const bar = document.getElementById(`progressbar_${index}`);
        const label = document.getElementById(`output_class_name_${index}`);
        if (bar) {
          bar.innerHTML = confidence + '%';
          bar.style.width = confidence + '%';
          bar.setAttribute("aria-valuenow", confidence);
        }
        if (label) {
          label.textContent = CLASS_NAMES[index];
        }
      });
    }
    prediction.dispose();
    imageFeatures.dispose();

    window.requestAnimationFrame(predictLoop);
  }
  else{
      await new Promise(r => setTimeout(r, 1000));
      predictLoop();
  }
}

function createPredictionUI() {
  const container = document.getElementById("prediction-container");
  container.innerHTML = "";
  const br = document.createElement("br");
  const inputtitle = document.createElement("h4");
  inputtitle.textContent = "Webcam Input";
  container.appendChild(inputtitle);
  VIDEO.setAttribute("style", "display: block; width: 100%; height: auto;");
  container.appendChild(VIDEO);
  container.appendChild(br);
  const title = document.createElement("h5");
  title.textContent = "Predictions";
  container.appendChild(title);
  CLASS_NAMES.forEach((name, index) => {
    const row = document.createElement("div");
    row.classList.add("prediction-row");

    const label = document.createElement("span");
    label.id = `output_class_name_${index}`;
    label.textContent = name;
    label.style.display = "inline-block";
    label.style.width = "120px";

    const progressWrapper = document.createElement("div");
    progressWrapper.classList.add("progress");
    progressWrapper.style.height = "20px";
    progressWrapper.style.flexGrow = "1";

    const bar = document.createElement("div");
    bar.id = `progressbar_${index}`;
    bar.className = "progress-bar bg-success";
    bar.setAttribute("role", "progressbar");
    bar.setAttribute("aria-valuenow", "0");
    bar.setAttribute("aria-valuemin", "0");
    bar.setAttribute("aria-valuemax", "100");
    bar.style.width = "0%";

    progressWrapper.appendChild(bar);
    row.appendChild(label);
    row.appendChild(progressWrapper);
    container.appendChild(row);
  });
}

function rebuildModel() {
  if (model) {
    model.dispose();
  }

  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [1024], units: 128, activation: 'relu' }));

  if (CLASS_NAMES.length === 2) {
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile({
      optimizer: 'adam',
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });
  } else {
    model.add(tf.layers.dense({ units: CLASS_NAMES.length, activation: 'softmax' }));
    model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
  }
}

