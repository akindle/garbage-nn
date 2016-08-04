using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace garbage
{
    public class MnistDataLoader
    {
        public int Columns { get; private set; }
        public bool Loaded;
        private readonly string root = @"C:\Users\alex\";

        public int Rows { get; private set; }
        public List<Network.DataSet> TestingData;
        public List<Network.DataSet> TrainingData;

        public MnistDataLoader()
        {
            Loaded = false;
        }

        public async Task Load()
        {
            var trainDataName = "train-images-idx3-ubyte.gz";
            var trainLabelName = "train-labels-idx1-ubyte.gz";
            var testDataName = "t10k-images-idx3-ubyte.gz";
            var testLabelName = "t10k-labels-idx1-ubyte.gz";
            TrainingData = await GetDataFromNames(root + trainDataName, root + trainLabelName);
            TestingData = await GetDataFromNames(root + testDataName, root + testLabelName);
            Loaded = true;
        }

        public async Task<List<Network.DataSet>> GetDataFromNames(string dataName, string labelName)
        {
            var datastream =
                new BigEndianBinaryReader(new GZipStream(File.OpenRead(dataName), CompressionMode.Decompress));
            var labelstream =
                new BigEndianBinaryReader(new GZipStream(File.OpenRead(labelName), CompressionMode.Decompress));
            if (datastream.ReadInt32() != 2051 || labelstream.ReadInt32() != 2049)
                throw new Exception("magic number mismatch");
            var size = await datastream.ReadInt32Async();
            var otherSize = await labelstream.ReadInt32Async();
            if (size != otherSize) throw new Exception("Size mismatch between data and labels");
            var results = new List<Network.DataSet>(size);
            Rows = await datastream.ReadInt32Async();
            Columns = await datastream.ReadInt32Async();
            for (var i = 0; i < size; i++)
            {
                var label = labelstream.ReadByte();
                // var buffer = new byte[Rows*Columns];
                // await datastream.ReadBytesAsync(buffer, 0, Rows*Columns);
                var buffer = datastream.ReadBytes(Rows*Columns);
                var data = CreateVector.Dense(buffer.Select(a => (double) a).ToArray());
                results.Add(new Network.DataSet(data, label, 10));
            }
            return results;
        }
    }
}