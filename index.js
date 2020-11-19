let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var happySamples=0, sadSamples=0, angrySamples=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobile_net_url = 'https://connor11son.github.io/model.json';
  const mobilenet = await tf.loadLayersModel(mobile_net_url);

  const layer = mobilenet.getLayer('Conv_1');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(4);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 1024, activation: 'relu'}),
      tf.layers.dense({ units: 512, activation: 'relu'}),
      tf.layers.dense({ units: 256, activation: 'relu'}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 4, activation: 'softmax'})
    ]
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 40,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}


function handleButton(elem){
	switch(elem.id){
		case "0":
			happySamples++;
			document.getElementById("happysamples").innerText = "Happy samples:" + happySamples;
			break;
		case "1":
			sadSamples++;
			document.getElementById("sadsamples").innerText = "Sad samples:" + sadSamples;
			break;
		case "2":
			angrySamples++;
			document.getElementById("angrysamples").innerText = "Angry samples:" + angrySamples;
			break;
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);

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
    switch(classId){
		case 0:
			predictionText = "You are happy.";
			break;
		case 1:
			predictionText = "You are sad.";
			break;
		case 2:
			predictionText = "You are angry.";
			break;
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}



init();
