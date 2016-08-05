using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace garbage
{
    public class SoftmaxLayer : Layer
    {
        public SoftmaxLayer(int inputSize, int outputSize, double eta, Layer nextLayer = null) : base(inputSize, outputSize, eta, nextLayer)
        {
        }

        internal override Matrix<double> ActivationFunction()
        {
            _biasesMatrix = CreateMatrix.DenseOfColumnVectors(
                Generate.LinearRange(1, Inputs.ColumnCount).Select(_ => _biases));
            var z = (Weights*Inputs + _biasesMatrix);
            var z_e = z.PointwiseExp();
            var sum = z_e.ColumnSums();
            var dividers = CreateMatrix.DenseOfRowVectors(Generate.LinearRange(1, z_e.RowCount).Select(_ => sum));
            return z_e.PointwiseDivide(dividers);
        }

        public override Matrix<double> Backpropagate(Matrix<double> labels)
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
                delta = (Outputs - labels);//.PointwiseMultiply(SigmoidPrime);
            }
            _biasesMatrix = CreateMatrix.DenseOfColumnVectors(
                Generate.LinearRange(1, Inputs.ColumnCount).Select(_ => _biases));
            var del_w = delta * Inputs.Transpose();
            var nb = delta.ReduceColumns((vector, doubles) => vector + doubles)
                .Divide(delta.ColumnCount);
            _biases = _biases - _eta / Inputs.ColumnCount * nb;
            Weights = Weights - _eta / Inputs.ColumnCount * del_w;
            return delta;
        }
    }
}
