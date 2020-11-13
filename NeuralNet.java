import java.util.Arrays;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

class NeuralNet {
  private int nbLayers, nbfeatures;
  private double[][] inputFeatures, catInputFeats, resultSet, dataset;
  private double[] outputWeights;
  private Neuron[][] hiddenLayers;
  private double outputBias;
  private double lR;
  public double[] labels;

  public NeuralNet(int layers, int inputSize, int nbunits, int[] catgSizes) {
    this.lR = 0.01;
    this.nbLayers = layers - 1;
    this.nbfeatures = inputSize;
    this.hiddenLayers = new Neuron[this.nbLayers][nbunits];
    
    for (int i = 0; i < this.nbLayers - 1; i++)
      for (int j = 0; j < nbunits; j++) {
        if (i == 0)
          hiddenLayers[i][j] = new Neuron(inputSize - catgSizes.length, catgSizes);
        else {
          int lastlayer = this.hiddenLayers[i - 1].length; 
          hiddenLayers[i][j] = new Neuron(lastlayer, catgSizes);
        }
      }
    
    this.hiddenLayers[this.nbLayers - 1] = new Neuron[1];

    for (int j = 0; j < 1; j++)
      this.hiddenLayers[this.nbLayers - 1][j] = new Neuron(nbunits, catgSizes);
  }
  
  private double[] convertDouble(String[] ins) {
    double[] res = new double[ins.length];
    for (int i = 0; i < ins.length; i++) {
      res[i] = Double.parseDouble(ins[i]);
    }
    return res;
  }

  public void readCSV(String filename, int datasize, int nbfeatures, int[] catFeaturesIdx) {
    String line = "";
    double[][] tmpDataset = new double[datasize][nbfeatures];
    this.labels = new double[datasize];

    int i = 0;
    try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
      while ((line = br.readLine()) != null) {
        if (i > 0)
          tmpDataset[i - 1] = convertDouble(line.split(";"));
        i++;
      }
    } catch (IOException e) {
      e.printStackTrace();
    }

    int k = 0, l = 0;
    
    this.dataset = new double[datasize][nbfeatures - catFeaturesIdx.length];
    this.catInputFeats = new double[datasize][catFeaturesIdx.length];
    int j;
    for (i = 0; i < datasize; i++) {
      for (j = 0; j < nbfeatures; j++) {
        
        if (!inVector(catFeaturesIdx, j)) {
          this.dataset[i][k] = tmpDataset[i][j];
          k++;
        } else {
          if (l == 6 || l==11)
            tmpDataset[i][j] += 1;
          
          if (l == 12) {
            if (tmpDataset[i][j] == 3)
              tmpDataset[i][j] = 1;
            else if (tmpDataset[i][j] == 6)
              tmpDataset[i][j] = 2;
            else if (tmpDataset[i][j] == 7)
              tmpDataset[i][j] = 3;
            else 
              tmpDataset[i][j] = 0;
          }

          this.catInputFeats[i][l] = tmpDataset[i][j];
          l++;
        }
      }
      this.labels[i] = tmpDataset[i][j];
      l = 0;
      k = 0;
    }
  }

  private Boolean inVector(int[] x, int k) {
    for (int i : x)
      if (k == i)
        return true;
    return false;
  }

  private void normalizeAux(int j) {
    double N = this.dataset.length, 
      mean = 0, 
      max = this.dataset[0][j], 
      min = this.dataset[0][j];
    
    for (int i = 0; i < this.dataset.length; i++) {
      double e = this.dataset[i][j]; 
      mean += e;
      if (e > max)
        max = e;
      if (e < min)
        min = e;
    }
    
    mean /= N;
    for (int i = 0; i < this.dataset.length; i++) {
      this.dataset[i][j] = (this.dataset[i][j] - min) / (max - min);
    }
  }
  
  public void normalizeZAux(int j) {
     double N = this.dataset.length, mean = 0;
     double sd = 0;
    
    for (int i = 0; i < this.dataset.length; i++) {
      double e = this.dataset[i][j]; 
      mean += e;
    }
    mean /= N;

    for (int i = 0; i < this.dataset.length; i++) {
      double e = this.dataset[i][j]; 
      sd += Math.pow(e - mean, 2);
    }
    sd = Math.sqrt((1.0 / N) * sd);
    
    for (int i = 0; i < this.dataset.length; i++) {
      this.dataset[i][j] = (this.dataset[i][j] - mean) / sd;
    }
  }


  public void normalizeSetMinMax() {
    for (int j = 0; j < this.dataset[0].length; j++) {
      normalizeAux(j);
    }
  }

  public void normalizeSetZscore() {
    for (int j = 0; j < this.dataset[0].length; j++) {
      normalizeZAux(j);
    }
  }
  
  public double[] feedForward(int indexBatch, int[] catgSizes) {
    double[] tmpFeatures = new double[this.nbfeatures - catgSizes.length], tmpVec;
    double[] tmpCatg = new double[catgSizes.length];

    for (int j = 0; j < this.nbfeatures - catgSizes.length; j++) {
      tmpFeatures[j] = this.dataset[indexBatch][j];
    }

    for (int j = 0; j < catgSizes.length; j++) {
      tmpCatg[j] = this.catInputFeats[indexBatch][j];
    }
    
    for (int row = 0; row < this.hiddenLayers.length; row++) {
      tmpVec = new double[this.hiddenLayers[row].length];
      
      for (int i = 0; i < this.hiddenLayers[row].length; i++) {
        Neuron n = this.hiddenLayers[row][i];
        double tmpx;
        if (row == 0)
          tmpx = n.getOutput(tmpFeatures, tmpCatg, catgSizes);
        else 
          tmpx = n.getOutput(tmpFeatures);
          
        tmpVec[i] = Matrix.sigmoid(tmpx);
        n.setOutput(tmpVec[i]);
      }
      tmpFeatures = tmpVec;
    }
    return tmpFeatures;
  }

  public void backPropagate(double[] label, int indexBatch) {
    for (int i = this.hiddenLayers.length - 1; i >= 0; i--) {
      double[] errors = new double[this.hiddenLayers[i].length];
      double[] biasErrors = new double[this.hiddenLayers[i].length];

      if (i != this.hiddenLayers.length - 1) {
        for (int j = 0; j < this.hiddenLayers[i].length; j++) {
            double error = 0;
            for (Neuron n : this.hiddenLayers[i + 1]) {
              error += n.getWeight(j) * n.getDelta();
            }
            errors[j] = error;
        }
      } else {
        for (int j = 0; j < this.hiddenLayers[i].length; j++) {
          errors[j] = label[j] - this.hiddenLayers[i][j].getOutput();
        }
      }
      
      for (int k = 0; k < this.hiddenLayers[i].length; k++) {
        Neuron n = this.hiddenLayers[i][k];
        n.setDelta(errors[k] * Matrix.sigmoidD(n.getOutput()));
      }
    }
  }

  public void updateWeights(int indexBatch) {
    double[] tmpFeatures = new double[this.nbfeatures - this.catInputFeats[indexBatch].length];
    double[] tmpCatg = new double[catInputFeats[indexBatch].length];

    for (int j = 0; j < this.nbfeatures - catInputFeats[indexBatch].length; j++)
      tmpFeatures[j] = this.dataset[indexBatch][j];
    
    for (int j = 0; j < this.catInputFeats[indexBatch].length; j++)
      tmpCatg[j] = this.catInputFeats[indexBatch][j];
    
    for (int i = 0; i < this.hiddenLayers.length; i++) {
      double[] inputs = tmpFeatures;

      if (i != 0) {
        inputs = new double[this.hiddenLayers[i -1].length];

        for (int j = 0; j < this.hiddenLayers[i - 1].length; j++) {
          inputs[j] = this.hiddenLayers[i - 1][j].getOutput();
        }
      } 

      for (Neuron n : this.hiddenLayers[i]) {
        for (int j = 0; j < inputs.length; j++) {
          n.updateWeight(j, this.lR * n.getDelta() * inputs[j]);
        }
        n.updateBias(this.lR * n.getDelta());
      }

      if (i == 0) {
        for (Neuron n : this.hiddenLayers[i])
          for (int j = 0; j < tmpCatg.length; j++) {
            n.updateWeight(j, (int) tmpCatg[j] - 1, this.lR * n.getDelta() * tmpCatg[j]); 
          }
      }
    }
  }

  public void printDataset() { 
    for (double[] row : this.dataset) {
      System.out.println(Arrays.toString(row));
    }
  }
}
