let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var rockSamples = 0,
    paperSamples = 0,
    scissorsSamples = 0,
    nullSamples = 0;
let isPredicting = false;

async function loadMobilenet() {
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

function buildModel() {
    model = tf.sequential({
        layers: [
            tf.layers.flatten(
                {inputShape: mobilenet.outputs[0].shape.slice(1)}
            ),
            tf.layers.dense(
                {units: 100, activation: 'relu'}
            ),
            tf.layers.dense(
                {units: 4, activation: 'softmax'}
            )
        ]
    });
  
}
async function train() {
    let weights=null;
    if(model){
      weights=model.getWeights();
    }else{
      buildModel();
    }
    const optimizer = tf.train.adam(0.0001);
    model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
    //load weight
    if(weights)
      model.loadWeights(weights,0);
    //
    dataset.ys = null;
    dataset.encodeLabels(4);


    // 可视化配置
    let container = {
        name: 'My Model',
        tab: "训练",
        name: 'training',
        style: {
            height: '1000px'
        }
    };

    let metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    let fitCallbacks = tfvis.show.fitCallbacks(container, metrics);


    model.fit(dataset.xs, dataset.ys, {
        epochs: 10,
        callbacks: fitCallbacks
    });
}


function handleButton(elem) {
    switch (elem.id) {
        case "0": rockSamples++;
            document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
            break;
        case "1": paperSamples++;
            document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
            break;
        case "2": scissorsSamples++;
            document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
            break;
        case "3": nullSamples++;
            document.getElementById("nullssamples").innerText = "nulls samples:" + nullSamples;
            break;
    }
    label = parseInt(elem.id);
    const img = webcam.capture();
    dataset.addExample(mobilenet.predict(img), label);

}
function downLoadButton(elem) {
    model.save('downloads://rps');
}
async function loadLoadButton(elem) {
    model = await tf.loadLayersModel('data/rps.json');
    tfvis.show.modelSummary({
        name: 'model_sturct1',
        tab: '加载的模型结构'
    }, model);
}

async function predict() {
    while (isPredicting) {
        const predictedClass = tf.tidy(() => {
            const img = webcam.capture();
            const activation = mobilenet.predict(img);
            const predictions = model.predict(activation);
            return predictions.as1D().argMax();
        });
        const classId = (await predictedClass.data())[0];
        var predictionText = "";
        switch (classId) {
            case 0: predictionText = "I see Rock";
                break;
            case 1: predictionText = "I see Paper";
                break;
            case 2: predictionText = "I see Scissors";
                break;
            case 3: predictionText = "I see Null";
                break;
        }
        document.getElementById("prediction").innerText = predictionText;


        predictedClass.dispose();
        await tf.nextFrame();
    }
}


function doTraining() {
    train();
}

function startPredicting() {
    isPredicting = true;
    predict();
}

function stopPredicting() {
    isPredicting = false;
    predict();
}

async function init() {
    await webcam.setup();
    mobilenet = await loadMobilenet();
    // buildModel();
    mobilenet.summary()
    tf.tidy(() => mobilenet.predict(webcam.capture()));
    tfvis.visor().surface({tab: '训练数据', name: 'train_data'});


    tfvis.show.modelSummary({
        name: 'model_sturct_mobile',
        tab: 'mobileNet'
    }, mobilenet);
}


async function mergeNetwork(){
  let intermedia_output=mobilenet.output;
  intermedia_output=tf.layers.flatten().apply(intermedia_output);
  let final_output=model.apply(intermedia_output);

  let newModel=tf.model({
    inputs:mobilenet.inputs,
    outputs:final_output
  });
  return newModel;
}

init();
