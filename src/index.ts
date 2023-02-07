import * as tf from "@tensorflow/tfjs-node";

class NumberModel {
  model?: tf.Sequential;
  prediction?: any;
  constructor() {
    this.train().then(() => {
      this.test(10);
    });
  }
  async train() {
    // define the model for linear regression
    this.model = tf.sequential();
    this.model.add(
      tf.layers.dense({
        units: 1,
        inputShape: [1],
      })
    );

    // prepare the model for training
    this.model.compile({
      loss: "meanSquaredError",
      optimizer: "sgd",
    });

    const xs = tf.tensor1d([-1, 0, 1, 2, 3, 4]);
    const ys = tf.tensor1d([-3, -1, 1, 3, 5, 7]);

    // train
    await this.model.fit(xs, ys, {
      epochs: 500,
    });

    console.log("✅ Model Trained");
  }

  test(value: number) {
    if (!this.model) {
      console.error("Model is not defined yet!");
      return;
    }
    const output = this.model.predict(tf.tensor2d([value], [1, 1])) as any;

    console.log("Prediction output", Array.from(output.dataSync()));
  }
}

new NumberModel();
