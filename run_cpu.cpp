// g++ -O3 -o run_cpu run_cpu.cpp -lgmpxx -lgmp -fopenmp `pkg-config --cflags --libs opencv4`
#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <chrono>

using namespace std;
constexpr int IMG_SIZE = 65536;
constexpr int MAX_ITER = 10000;

int run_iter(const double x0, const double y0)
{
    double x = x0;
    double y = y0;
    for (int i = 0; i < MAX_ITER; i++)
    {
        const double x2 = x * x;
        const double y2 = y * y;
        const double hyp2 = x2 + y2;
        if (i == 0 && hyp2 < 1.0 / 16)
            return 0;
        if (hyp2 > 4)
            return 255;
        y = 2 * x * y + y0;
        x = x2 - y2 + x0;
    }
    return 0;
}

int main()
{
    const int size_half = IMG_SIZE / 2;
    const double zoom_lv = 2.0 / size_half;
    cv::Mat image = cv::Mat::ones(IMG_SIZE, IMG_SIZE, CV_8UC1) * 255;
    auto start = chrono::high_resolution_clock::now();
    int i, j;
    #pragma omp parallel for private(i, j)
    for (i = 0; i <= size_half; i++)
        for (j = 0; j < IMG_SIZE; j++)
        {
            const double y = (i - size_half) * zoom_lv;
            const double x = (j - size_half) * zoom_lv;
            const int ans = run_iter(x, y);
            image.at<uchar>(i, j) = ans;
            if (i > 0 && i < size_half)
                image.at<uchar>(IMG_SIZE - i, j) = ans;
        }
    auto stop = chrono::high_resolution_clock::now();
    cout << "CPU Time: " << chrono::duration_cast<chrono::microseconds>(stop - start).count() / 1e6 << " seconds" << endl;

    int i_min = IMG_SIZE;
    int i_max = 0;
    int j_min = IMG_SIZE;
    int j_max = 0;
    for (i = 0; i < IMG_SIZE; i++)
        for (j = 0; j < IMG_SIZE; j++)
            if (image.at<uchar>(i, j) <= 127)
            {
                i_min = min(i, i_min);
                i_max = max(i, i_max);
                j_min = min(j, j_min);
                j_max = max(j, j_max);
            }
    image = image(cv::Rect(j_min, i_min, j_max + 1 - j_min, i_max + 1 - i_min));
    cv::imwrite("out_cpu.png", image);
    return 0;
}
