using System;
using System.Collections.Generic;
using MathNet.Numerics;

namespace garbage
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var x = new MnistDataLoader();
            x.Load().Wait();
            var network = new Network(new List<int> {784, 300, 30, 10});
            while(true) network.StochasticGradientDescent(x.TrainingData, 300, 3, x.TestingData).Wait();
        }
    }
}