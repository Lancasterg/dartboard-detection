//
// Created by Jeaung on 11/12/18.
//
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include "opencv2/highgui/highgui.hpp"   //adjust import locations
#include "opencv2/imgproc/imgproc.hpp"    //depending on your machine setup
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>
#include "header/hough.h"

using namespace std;
using namespace cv;


Mat getYKernel() {
    Mat kernel(Size(3, 3), CV_32F);
    kernel.at<float>(0, 0) = -1;
    kernel.at<float>(0, 1) = -2;
    kernel.at<float>(0, 2) = -1;
    kernel.at<float>(1, 0) = 0;
    kernel.at<float>(1, 1) = 0;
    kernel.at<float>(1, 2) = 0;
    kernel.at<float>(2, 0) = 1;
    kernel.at<float>(2, 1) = 2;
    kernel.at<float>(2, 2) = 1;
    return kernel;
}

Mat getXKernel() {
    Mat kernel(Size(3, 3), CV_32F);
    kernel.at<float>(0, 0) = -1;
    kernel.at<float>(0, 1) = 0;
    kernel.at<float>(0, 2) = 1;
    kernel.at<float>(1, 0) = -2;
    kernel.at<float>(1, 1) = 0;
    kernel.at<float>(1, 2) = 2;
    kernel.at<float>(2, 0) = -1;
    kernel.at<float>(2, 1) = 0;
    kernel.at<float>(2, 2) = 1;
    return kernel;
}

int ***allocate3DArray(int x, int y, int z) {
    int ***the_array = new int **[x];
    for (int i(0); i < x; ++i) {
        the_array[i] = new int *[y];

        for (int j(0); j < y; ++j) {
            the_array[i][j] = new int[z];

            for (int k(0); k < z; ++k) {
                the_array[i][j][k] = 0;
            }
        }
    }
    return the_array;
}


void display(const string &name, const Mat &src) {
    cv::Mat dst = src.clone();
    normalize(dst, dst, 0, 1, NORM_MINMAX);
    imshow(name, dst);
    waitKey(0);
}

void hough(string imageName, Mat magn, Mat dir, int threshold, int minRadius, int maxRadius) {
    int ***vote = allocate3DArray(magn.rows, magn.cols, maxRadius);

    for (int i = 0; i < magn.rows; i++) {
        for (int j = 0; j < magn.cols; j++) {
            float mag = magn.at<float>(i, j);
            float ori = dir.at<float>(i, j);

            if (abs(mag - 255) < 1e-4) {
                for (int r = minRadius; r < maxRadius; r++) {
                    int x0 = (int) (i - (r) * sin(ori));
                    int y0 = (int) (j - (r) * cos(ori));
                    if (x0 >= 0 && x0 < magn.rows && y0 >= 0 && y0 < magn.cols) {
                        vote[x0][y0][r] += 1;
                    }
                    x0 = (int) (i + (r) * sin(ori));
                    y0 = (int) (j + (r) * cos(ori));
                    if (x0 >= 0 && x0 < magn.rows && y0 >= 0 && y0 < magn.cols) {
                        vote[x0][y0][r] += 1;
                    }
                }
            }
        }
    }

    Mat hough_space(magn.rows, magn.cols, CV_32F);

    for (int x = 0; x < magn.rows; x++) {
        for (int y = 0; y < magn.cols; y++) {
            for (int r = minRadius; r < maxRadius; r++) {
                hough_space.at<float>(x, y) += vote[x][y][r];
            }
        }
    }

    display("hough space", hough_space);

    Mat image = imread(imageName);

    for (int x = 0; x < magn.rows; x++) {
        for (int y = 0; y < magn.cols; y++) {
            for (int r = minRadius; r < maxRadius; r++) {
                if (vote[x][y][r] > threshold) {
                    printf("%d %d %d %d \n", x, y, r, vote[x][y][r]);
                }

                if (vote[x][y][r] > threshold) {
                    Point center(y, x);
                    circle(image, center, r, Scalar(255, 0, 0));
                }
            }
        }
    }


    imshow("detected circles", image);
    waitKey(0);
}

Mat gradMagnitude(Mat input) {
    Mat gradXKernel = getXKernel();
    Mat gradYKernel = getYKernel();

    Mat result(input.rows, input.cols, CV_32F);

    int depth = 1;

    // now we can do the convolution
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            int originX = i - depth;
            int originY = j - depth;

            if (i >= depth && i < input.rows - depth && j >= depth && j < input.cols - depth) {
                float sumX = 0, sumY = 0;

                for (int x = 0; x < gradXKernel.rows; x++) {
                    for (int y = 0; y < gradXKernel.cols; y++) {
                        sumX += gradXKernel.at<float>(x, y) * input.at<float>(originX + x, originY + y);
                        sumY += gradYKernel.at<float>(x, y) * input.at<float>(originX + x, originY + y);
                    }
                }

                result.at<float>(i, j) = sqrt(sumX * sumX + sumY * sumY);
            }
        }
    }

//    normalize(result, result, 0, 1, NORM_MINMAX);
//    imshow("gradient magnitude", result);
//    waitKey(0);

    return result;
}

Mat gradDirection(Mat input) {
    Mat gradXKernel = getXKernel();
    Mat gradYKernel = getYKernel();

    Mat result(input.rows, input.cols, CV_32F);

    int depth = 1;

    Mat dx(input.rows, input.cols, CV_32F);
    Mat dy(input.rows, input.cols, CV_32F);

    // now we can do the convolution
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            int originX = i - depth;
            int originY = j - depth;

            if (i >= depth && i < input.rows - depth && j >= depth && j < input.cols - depth) {
                float sumX = 0, sumY = 0;

                for (int x = 0; x < gradXKernel.rows; x++) {
                    for (int y = 0; y < gradXKernel.cols; y++) {
                        sumX += gradXKernel.at<float>(x, y) * input.at<float>(originX + x, originY + y);
                        sumY += gradYKernel.at<float>(x, y) * input.at<float>(originX + x, originY + y);
                    }
                }

                dx.at<float>(i, j) = sumX;
                dy.at<float>(i, j) = sumY;
                result.at<float>(i, j) = atan2(sumY, sumX);
            }
        }
    }

    display("dx", dx);
    display("dy", dy);

//    normalize(result, result, 0, 1, NORM_MINMAX);
//    imshow("gradient direction", result);
//    waitKey(0);

    return result;
}

void thresholdMag(Mat &image, int threshold) {
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image.at<float>(i, j) > threshold) {
                image.at<float>(i, j) = 255;
            } else {
                image.at<float>(i, j) = 0;
            }
        }
    }
}

int main(int argc, char **argv) {
    string imageName = "img/dart0.jpg";
    Mat image = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);

    Mat image2, blur_img;
    // convert to CV_32F
    image.convertTo(image2, CV_32F);

    GaussianBlur(image2, blur_img, Size(5, 5), 0, 0);

    Mat mag = gradMagnitude(blur_img);
    display("gradient magnitude", mag);

    thresholdMag(mag, 120);
    display("threshold magnitude", mag);

    Mat dir = gradDirection(blur_img);
    display("gradient direction", dir);

    hough(imageName, mag, dir, 15, 40, 50);
}
