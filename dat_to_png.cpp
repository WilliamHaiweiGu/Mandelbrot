#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>


using namespace std;

constexpr unsigned int W_PIX = 65536;
constexpr unsigned int H_PIX = 65536;
const string DATA_FILE_NAME = "out.dat";

constexpr unsigned int H_PIX_HALF = H_PIX / 2;

int main()
{
    ifstream file(DATA_FILE_NAME, ios::binary | ios::in);
    
    if (!file) {
        std::cerr << "Unable to open " << DATA_FILE_NAME << endl;
        return 1;
    }   
    
    cv::Mat image = cv::Mat::ones(W_PIX, H_PIX, CV_8UC1) * 255;
    for (int i = 0; i <= H_PIX_HALF; i++)
        for (int j = 0; j < H_PIX; j++)
        {
            unsigned char pix;
            file.read(reinterpret_cast<char*>(&pix), 1);
            image.at<unsigned char>(i, j) = pix;
            if (i > 0 && i < H_PIX_HALF)
                image.at<unsigned char>(H_PIX - i, j) = pix;
        }
    file.close();
    int i_min = H_PIX;
    int i_max = 0;
    int j_min = W_PIX;
    int j_max = 0;
    for (int i = 0; i < H_PIX; i++)
        for (int j = 0; j < W_PIX; j++)
            if (image.at<unsigned char>(i, j) < static_cast<unsigned char>(255))
            {
                i_min = min(i, i_min);
                i_max = max(i, i_max);
                j_min = min(j, j_min);
                j_max = max(j, j_max);
            }
    image = image(cv::Rect(j_min, i_min, j_max + 1 - j_min, i_max + 1 - i_min));
    cv::imwrite("out.png", image);
    return 0;
}
