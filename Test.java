import java.util.Arrays;

public class Test {
  public static void main(String[] args) {
    int[] catgSizes = {4, 3, 3, 3};
    int[] catgIdx = {2, 6, 10, 12};
    int fp = 0, fn = 0, tp = 0, tn = 0;

    NeuralNet n = new NeuralNet(3, 13, 10, catgSizes);
    n.readCSV("./heart_disease_dataset.csv", 303, 13, catgIdx);
    n.normalizeSetZscore();
    int i = 1, epoch = 200, batch = 4;
    int k = 0;

    // training
    while (i++ < epoch) {
      int j = 0;
      int batchCount = 0;

      while (j < 101) {
        double[] target = {n.labels[j]};

        n.feedForward(j, catgSizes);
        n.backPropagate(target, j);
        
        if (batchCount == batch) {
          n.updateWeights(j);
          batchCount = 0;
        }
        j++;
        batchCount++;
      }

      j = 202;
      while (j < 303) {
        double[] target = {n.labels[j]};

        n.feedForward(j, catgSizes);
        n.backPropagate(target, j);
        
        if (batchCount == batch) {
          n.updateWeights(j);
          batchCount = 0;
        }
        j++;
        batchCount++;
      }
    }
    //validation

    // System.out.println(n.feedForward(168, catgSizes)[0]);
    // System.out.println(n.feedForward(201, catgSizes)[0]);
    // System.out.println(n.feedForward(105, catgSizes)[0]);
    // System.out.println(n.feedForward(142, catgSizes)[0]);
    
    i = 101;
    while (i < 202) {
      double x = n.feedForward(i, catgSizes)[0] < 0.5 ? 0 : 1;
      double label = n.labels[i];

      if (x == 1 && x != label)
        fp++;
      if (x == 1 && x == label)
        tp++;
      if (x == 0 && x != label)
        fn++;
      if (x == 0 && x == label)
        tn++;
      i++;
    }
    
    System.out.println("True Positives: " + tp);
    System.out.println("True Negatives: " + tn);
    System.out.println("False Positives: " + fp);
    System.out.println("False Negatives: " + fn);
    System.out.println("Accuracy: " + ((tp + tn) / 101.0) * 100.0);
  }
}

