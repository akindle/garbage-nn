﻿using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using garbage;

namespace garbage_wpf
{
    /// <summary>
    ///     Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly MnistDataLoader _data;
        private readonly Network network;

        public MainWindow()
        {
            InitializeComponent();

            _data = new MnistDataLoader();
            network = new Network(new List<int> {784, 300, 10});
        }

        private BitmapSource ImageFromData(Network.DataSet data)
        {
            return BitmapSource.Create(_data.Columns, _data.Rows, 96, 96, PixelFormats.Gray8, BitmapPalettes.Gray256,
                data.Data.Select(a => (byte) a).ToArray(), _data.Columns*sizeof (byte));
        }

        private void DrawTestData(Network.DataSet data)
        {
            Display.Width = _data.Columns;
            Display.Height = _data.Rows;
            Display.Source = ImageFromData(data);
            DisplayLabel.Content = $"Label: {data.Label.MaximumIndex()}";
            if (data.PredictedLabel != null)
            {
                DisplayLabel.Content += $"\r\nPredicted label: {data.PredictedLabel.MaximumIndex()}";
            }
        }

        private async void Iterate_OnClick(object sender, RoutedEventArgs e)
        {
            if (!Iterate.Content.Equals("Iterate")) return;
            if (!_data.Loaded)
            {
                Iterate.Content = "Loading...";
                await _data.Load();
                DrawTestData(_data.TestingData[10]);
            }
            Iterate.Content = "Iterating...";
            for (var i = 0; i < 10; i++)
            {
                await network.StochasticGradientDescent(_data.TrainingData, 1, 300, 3, _data.TestingData);
                DrawTestData(_data.TestingData[10]);
                Iterate.Content = $"{i} iterations...";
            }
            Iterate.Content = "Iterate";
        }
    }
}