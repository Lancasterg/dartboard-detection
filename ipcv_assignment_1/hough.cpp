//
// Created by Jeaung on 11/12/18.
//
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include "opencv2/highgui/highgui.hpp"   //adjust import locations
#include "opencv2/imgproc/imgproc.hpp"    //depending on your machine setup
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

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

void hough(Mat magn, Mat dir, int threshold, int minRadius, int maxRadius) {
    int ***vote = allocate3DArray(magn.rows, magn.cols, maxRadius);

    for (int i = 0; i < magn.rows; i++) {
        for (int j = 0; j < magn.cols; j++) {
            float mag = magn.at<float>(i, j);
            float ori = dir.at<float>(i, j);

            if (mag > 30) {
                for (int r = minRadius; r < maxRadius; r++) {
                    int x0 = (int) (i - r * cos(ori));
                    int y0 = (int) (j - r * sin(ori));
                    if (x0 >= 0 && x0 < magn.rows && y0 >= 0 && y0 < magn.cols) {
                        vote[x0][y0][r] += 1;
                    }

                    x0 = (int) (i + r * cos(ori));
                    y0 = (int) (j + r * sin(ori));
                    if (x0 >= 0 && x0 < magn.rows && y0 >= 0 && y0 < magn.cols) {
                        vote[x0][y0][r] += 1;
                    }
                }
            }
        }
    }

    Mat result(magn.rows, magn.cols, CV_32F);

    for (int x = 0; x < magn.rows; x++) {
        for (int y = 0; y < magn.cols; y++) {
            int sumRadius = 0;
            for (int r = minRadius; r < maxRadius; r++) {
                sumRadius += vote[x][y][r];
            }

            result.at<float>(x, y) = sumRadius;
        }
    }

    normalize(result, result, 0, 1, NORM_MINMAX);

//    for (int x = 0; x < magn.rows; x++) {
//        for (int y = 0; y < magn.cols; y++) {
//            for (int r = minRadius; r < maxRadius; r++) {
//                if (vote[x][y][r] > 4) {
//                    printf("%d %d %d %d \n", x, y, r, vote[x][y][r]);
//                }
//
//                if (vote[x][y][r] > threshold) {
//                    Point center(x, y);
//                    circle(result, center, r, 255);
//                }
//            }
//        }
//    }

    imshow("hough", result);
    waitKey(0);
    result.release();
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

                result.at<float>(i, j) = -atan2(sumY, sumX);
            }
        }
    }

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

//    imshow("threshold", image);
//    waitKey(0);
}

int main(int argc, char **argv) {
    char *imageName = argv[1];

    Mat image = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);

    if (argc != 2 || !image.data) {
        printf(" No image data \n ");
        return -1;
    }

    Mat image2;
    image.convertTo(image2, CV_32F);

    Mat mag = gradMagnitude(image2);

    Mat dir = gradDirection(image2);

    thresholdMag(mag, 200);

    hough(mag, dir, 5, 20, 100);

    return 0;
}
