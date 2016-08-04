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
        public static int SGD(List<DataSet> trainingData, int miniBatchSize, List<DataSet> testData, Layer input)
        {
            var rand = new Random();
            var shuffled = trainingData.OrderBy(a => rand.Next());
            var miniBatches =
                Generate.LinearRange(0, trainingData.Count, miniBatchSize)
                    .Select(k => shuffled.Skip((int) k).Take(miniBatchSize).ToList())
                    .Select(a => new Minibatch(a));

            foreach (var batch in miniBatches)
            {
                input.Feedforward(batch.inputMatrix);
                input.Backpropagate(batch.labelMatrix);
            }

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

        public class Layer
        {
            private readonly double _eta;

            private Matrix<double> __sp;
            private Vector<double> _biases;
            private Matrix<double> _biasesMatrix;
            private Matrix<double> _inputs;
            private readonly Layer _nextLayer;

            public Layer(int inputSize, int outputSize, double eta, Layer nextLayer = null)
            {
                _biases = CreateVector.Random<double>(outputSize);
                Weights = CreateMatrix.Random<double>(outputSize, inputSize);
                _eta = eta;
                _nextLayer = nextLayer;
            }

            public Matrix<double> Weights { get; private set; }

            public Matrix<double> Outputs { get; set; }

            public Matrix<double> Inputs
            {
                get { return _inputs; }
                set
                {
                    _inputs = value;
                    Outputs = ActivationFunction();
                }
            }

            private Matrix<double> _sigmoidPrime => __sp ?? (__sp = Outputs.PointwiseMultiply(1 - Outputs));

            public Matrix<double> Feedforward(Matrix<double> input)
            {
                Inputs = input;
                return _nextLayer != null ? _nextLayer.Feedforward(Outputs) : Outputs;
            }

            public Matrix<double> Cost(Matrix<double> labels)
            {
                return (Outputs - labels).PointwiseMultiply(_sigmoidPrime);
            }

            public Matrix<double> ActivationFunction()
            {
                _biasesMatrix = CreateMatrix.DenseOfColumnVectors(
                    Generate.LinearRange(1, Inputs.ColumnCount).Select(_ => _biases));
                return Sigmoid(Weights*Inputs + _biasesMatrix);
            }

            public Matrix<double> Backpropagate(Matrix<double> labels)
            {
                Matrix<double> delta;
                if (_nextLayer != null)
                {
                    delta =
                        _nextLayer.Weights.TransposeThisAndMultiply(_nextLayer.Backpropagate(labels))
                            .PointwiseMultiply(_sigmoidPrime);
                }
                else
                {
                    delta = (Outputs - labels).PointwiseMultiply(_sigmoidPrime);
                }
                _biasesMatrix = CreateMatrix.DenseOfColumnVectors(
                    Generate.LinearRange(1, Inputs.ColumnCount).Select(_ => _biases));
                var del_w = delta*Inputs.Transpose();
                var nb = delta.ReduceColumns((vector, doubles) => vector + doubles)
                    .Divide(delta.ColumnCount);
                _biases = _biases - _eta/Inputs.ColumnCount*nb;
                Weights = Weights - _eta/Inputs.ColumnCount*del_w;
                return delta;
            }


            public static Vector<double> Sigmoid(Vector<double> z)
            {
                return z.Map(SpecialFunctions.Logistic);
            }

            public static Matrix<double> Sigmoid(Matrix<double> z)
            {
                return z.Map(SpecialFunctions.Logistic);
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

        public static async Task<int> SGDAsync(List<DataSet> trainingData, int i, List<DataSet> testingData, Layer input)
        {
            return await Task.Run(() => SGD(trainingData, i, testingData, input));
        }
    }
}