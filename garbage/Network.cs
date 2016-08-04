using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace garbage
{
    public class Network
    {
        public static void SGD(List<DataSet> trainingData, int miniBatchSize, List<DataSet> testData, Layer input)
        {
            var rand = new Random();
            var shuffled = trainingData.OrderBy(a => rand.Next()).Take(miniBatchSize).ToList();
           // var miniBatches =
           //     Generate.LinearRange(0, trainingData.Count, miniBatchSize)
           //         .Select(k => shuffled.Skip((int) k).Take(miniBatchSize).ToList())
           //         .Select(a => new Minibatch(a));
            var inputMatrix = CreateMatrix.DenseOfColumnVectors(shuffled.Select(a => a.Data));
            var labelMatrix = CreateMatrix.DenseOfColumnVectors(shuffled.Select(a => a.Label));
            input.Feedforward(inputMatrix);
            input.Backpropagate(labelMatrix);
            //foreach (var batch in miniBatches)
            //{
            //    input.Feedforward(batch.inputMatrix);
            //    input.Backpropagate(batch.labelMatrix);
            //}
        }

        public static int Evaluate(List<DataSet> testData, Layer input)
        {
            var testMatrix = CreateMatrix.DenseOfColumnVectors(testData.Select(a => a.Data));
            var outputs = input.Feedforward(testMatrix);
            foreach (var x in testData.Zip(outputs.EnumerateColumns(), (d, v) => new {d, v}))
            {
                x.d.PredictedLabel = x.v;
            }
            var matrixResults = outputs.EnumerateColumns().Select(a => a.MaximumIndex());
            var pass = matrixResults.Zip(testData.Select(a => a.Label.MaximumIndex()), (a, b) => a == b).Count(a => a);
            return pass;
        }

        public class DataSet
        {
            public readonly Vector<double> Data;
            public readonly Vector<double> Label;
            public Vector<double> PredictedLabel;

            public DataSet(Vector<double> data, int label, int labelDimensionality)
            {
                Data = data;
                Label = CreateVector.Dense<double>(labelDimensionality);
                Label[label] = 1;
            }
        }

        public class Minibatch
        {
            public readonly Matrix<double> inputMatrix;
            public readonly List<DataSet> inputs;
            public readonly Matrix<double> labelMatrix;

            public Minibatch(List<DataSet> inputs)
            {
                this.inputs = inputs;
                inputMatrix = CreateMatrix.DenseOfColumnVectors(inputs.Select(a => a.Data));
                labelMatrix = CreateMatrix.DenseOfColumnVectors(inputs.Select(a => a.Label));
            }
        }

        public static async Task SGDAsync(List<DataSet> trainingData, int i, List<DataSet> testingData, Layer input)
        {
            await Task.Run(() => SGD(trainingData, i, testingData, input));
        }
    }
}