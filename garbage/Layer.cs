using System.Linq;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace garbage
{
    public class Layer
    {
        internal readonly double _eta;
        internal readonly Layer _nextLayer;

        internal Matrix<double> __sp;
        internal Vector<double> _biases;
        internal Matrix<double> _biasesMatrix;
        internal Matrix<double> _inputs;

        public Layer(int inputSize, int outputSize, double eta, Layer nextLayer = null)
        {
            _biases = CreateVector.Random<double>(outputSize);
            Weights = CreateMatrix.Random<double>(outputSize, inputSize);
            _eta = eta;
            _nextLayer = nextLayer;
        }

        public Matrix<double> Weights { get; internal set; }

        public Matrix<double> Outputs { get; set; }

        internal Matrix<double> Inputs
        {
            get { return _inputs; }
            set
            {
                _inputs = value;
                Outputs = ActivationFunction();
                __sp = Outputs.PointwiseMultiply(1 - Outputs);
            }
        }

        internal Matrix<double> SigmoidPrime => __sp ?? (__sp = Outputs.PointwiseMultiply(1 - Outputs));

        public Matrix<double> Feedforward(Matrix<double> input)
        {
            Inputs = input;
            return _nextLayer != null ? _nextLayer.Feedforward(Outputs) : Outputs;
        }

        internal virtual Matrix<double> ActivationFunction()
        {
            _biasesMatrix = CreateMatrix.DenseOfColumnVectors(
                Generate.LinearRange(1, Inputs.ColumnCount).Select(_ => _biases));
            return Sigmoid(Weights*Inputs + _biasesMatrix);
        }

        public virtual Matrix<double> Backpropagate(Matrix<double> labels)
        {
            Matrix<double> delta;
            if (_nextLayer != null)
            {
                delta =
                    _nextLayer.Weights.TransposeThisAndMultiply(_nextLayer.Backpropagate(labels))
                        .PointwiseMultiply(SigmoidPrime);
            }
            else
            {
                // cost function
                delta = (Outputs - labels).PointwiseMultiply(SigmoidPrime);
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

        private static Matrix<double> Sigmoid(Matrix<double> z)
        {
            return z.Map(SpecialFunctions.Logistic);
        }
    }
}