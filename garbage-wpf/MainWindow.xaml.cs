using System;
using System.Collections.Generic;
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
        private readonly Layer _input;

        public MainWindow()
        {
            InitializeComponent();

            _data = new MnistDataLoader();
            _input = new Layer(784, 100, 1, new SoftmaxLayer(100, 10, 1));
            //_input = new Layer(784, 10, 1);
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
                ResultSelector.Minimum = 0;
                ResultSelector.Maximum = _data.TestingData.Count;
                DrawTestData(_data.TestingData[10]);
                Iterate.Content = "Iterate";
                return;
            }
            Iterate.Content = "Iterating...";
            for (var i = 0; i < Int32.Parse(Iterations.Text); i++)
            {
                await Network.SGDAsync(_data.TrainingData, 3000, _data.TestingData, _input);
                Iterate.Content = $"{i} iterations...";
            //    var results = _data.TestingData.Select(a => a.Label)
            //        .Zip(_data.TestingData.Select(a => a.PredictedLabel), (b, a) => b.MaximumIndex() == a.MaximumIndex()).ToList();
                if (i%10 == 0)
                {
                    var res = await Task.Run(() => Network.Evaluate(_data.TestingData, _input));
                    PerformanceLabel.Content = $"{res} / {10000}";
                    ResultSelector_OnValueChanged(null, null);
                }
            }
            Iterate.Content = "Iterate";
        }

        private void ResultSelector_OnValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (_data.Loaded)
            {
                var drawSource = _data.TestingData.Where(a => !a.Predicted).ToList();
                ResultSelector.Maximum = drawSource.Count;
                DrawTestData(drawSource[(int)ResultSelector.Value]);
            }
        }
    }
}