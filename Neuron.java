import java.util.Arrays;

class Neuron {
  private String id;
  private double bias, output;
  private double[] weights;
  private double[][] catgWeights; 
  private double delta;

  public Neuron(int n, int[] catgSizes) {
    this.initWeights(n);
    this.setBias();
    this.catgWeights = new double[catgSizes.length][];

    for (int i = 0; i < catgSizes.length; i++)
        this.catgWeights[i] = this.initWeightsCat(catgSizes[i]);
  }

  private void initWeights(int n) {
    this.weights = new double[n];
    for (int i = 0; i < n; i++)
      this.weights[i] = Matrix.RandNum(-1, 1);
  }
  
  private double[] initWeightsCat(int n) {
    double[] weights = new double[n];
    for (int i = 0; i < n; i++)
      weights[i] = Matrix.RandNum(-1, 1);
    return weights;
  }

  private void setBias() {
    this.bias = Matrix.RandNum(-1, 1);
  }

  public void updateBias(double x) {
    this.bias += x;
  }

  public double getBias() {
    return this.bias;
  }
  
  static public double[] onehotEncode(int n, int k) {
    double[] res = new double[n];
    for (int i = 0; i < n; i++)
      res[i] = 0;
    if (k != 0 && (k - 1) != n)
      res[k - 1] = 1;
    return res;
  }
  
  public double getOutput(double[] inputs, double[] catgFeatures, int[] catgLengths) {
    double sum = 0;

    for (int i = 0; i < catgFeatures.length; i++) {
      double encode[] = Neuron.onehotEncode(catgLengths[i], (int) catgFeatures[i]);
      sum += Matrix.dot(encode, this.catgWeights[i]);
    }

    return Matrix.dot(this.weights, inputs) + sum + this.bias;
  }
  
  public double getOutput(double[] inputs) {
    return Matrix.dot(inputs, this.weights) + this.bias;
  }

  public double[] getWeights() {
    return this.weights;
  }

  public double getWeight(int i) {
    return this.weights[i];
  }
  
  public void updateWeight(int i, double x) {
    this.weights[i] += x;
  }
  
  public void updateWeight(int i, int j, double x) {
    if (j < 0)
      return;
    this.catgWeights[i][j] += x;
  }

  public double getDelta() {
    return this.delta;
  }

  public void setDelta(double d) {
    this.delta = d;
  }
  
  public double getOutput() {
    return this.output;
  }

  public void setOutput(double o) {
    this.output = o;
  }

  public void printCatMat(int i) {
    System.out.println(Arrays.toString(this.catgWeights[i]));
  }
  
  public void printMatrix() {
    System.out.println(Arrays.toString(this.weights));
  }
}
