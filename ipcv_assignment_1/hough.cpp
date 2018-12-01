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

/**
 * Create a vertical sobel kernel
 * @return
 */
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

/**
 * Create horizontal sobel kernel
 * @return
 */
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

/**
 * Allocate 3D array for circle hough transform
 * @param x
 * @param y
 * @param z
 * @return the_array
 */
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

/**
 * Allocate 2D array for hough line transform
 * @param x
 * @param y
 * @return
 */
int **allocate2DArray(int x, int y) {
    int **the_array = new int *[x];
    for (int i(0); i < x; ++i) {
        the_array[i] = new int[y];

        for (int j(0); j < y; ++j) {
            the_array[i][j] = 0;
        }
    }
    return the_array;
}

/**
 * Normalise, and then display an image
 * @param name
 * @param src
 */
void display(const string &name, const Mat &src) {
    cv::Mat dst = src.clone();
    normalize(dst, dst, 0, 1, NORM_MINMAX);
    imshow(name, dst);
    waitKey(0);
    dst.release();
}

/**
 * Circle hough transform
 *
 * 1. Blur the image and apply sobel filter
 * 2. Get gradient direction
 * 3. Loop through all pixels
 * If pixel is above threshold
 *      3a. loop through all values of Radius
 *      3b. calculate hough space value using the gradient direction
 *      3c. Increment matrix at position
 * 4. Create hough space by flattening the 3rd dimension
 * 5. Add detected circles to list of vec3f [x,y,r]
 * @param src
 * @param threshold
 * @param minRadius
 * @param maxRadius
 * @return
 */
vector<Vec3f> hough_circle(const Mat &src, int threshold, int minRadius, int maxRadius) {
    Mat gray_image;
    cvtColor(src, gray_image, CV_BGR2GRAY);

    Mat image, blur_img;
    gray_image.convertTo(image, CV_32F);

    GaussianBlur(image, blur_img, Size(5, 5), 0, 0);

    Mat magn = gradMagnitude(blur_img);
//    display("gradient magnitude", magn);

    thresholdMag(magn, 120);
//    display("threshold magnitude", magn);

    Mat dir = gradDirection(blur_img);
//    display("gradient direction", dir);

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
//    display("hough space", hough_space);

    vector<Vec3f> circles;

    for (int x = 0; x < magn.rows; x++) {
        for (int y = 0; y < magn.cols; y++) {
            for (int r = minRadius; r < maxRadius; r++) {
                if (vote[x][y][r] > threshold) {
                    circles.insert(circles.end(), Vec3f(x, y, r));
//                    Point center(y, x);
//                    printf("%d %d %d %d \n", x, y, r, vote[x][y][r]);
//                    circle(src, center, r, Scalar(255, 0, 0));
                }
            }
        }
    }

//    imshow("detected circles", src);
//    waitKey(0);

    return circles;
}

/**
 * Hough line transform
 *
 * 1. Blur the image and apply sobel filter
 * 2. Get gradient direction
 * 3. Loop through all pixels
 * If pixel value is above threshold
 *      3a. detect line using gradient direction
 *      3b. if value fits in accumulator matrix
 *      3c. increment matrix at position [rho, degrees]
 * 4. All lines above threshold are kept
 * @param src
 * @param threshold
 * @param delta
 * @return
 */
vector<Vec2f> hough_line(const Mat &src, float threshold, int delta) {
    Mat gray_image;
    cvtColor(src, gray_image, CV_BGR2GRAY);

    Mat image, blur_img;
    gray_image.convertTo(image, CV_32F);

    GaussianBlur(image, blur_img, Size(5, 5), 0, 0);

    Mat magn = gradMagnitude(blur_img);
//    display("gradient magnitude", magn);

    thresholdMag(magn, 120);
//    display("threshold magnitude", magn);

    Mat dir = gradDirection(blur_img);
//    display("gradient direction", dir);

    int diag = (int) floor(sqrt(magn.rows * magn.rows + magn.cols * magn.cols));

    int **votes = allocate2DArray(diag, 361);
    Mat hough_space(diag, 361, CV_32F);

    for (int i = 0; i < magn.rows; i++) {
        for (int j = 0; j < magn.cols; j++) {
            float mag = magn.at<float>(i, j);
            float ori = dir.at<float>(i, j);
            int oriDegree = (int) (ori * 180 / M_PI);   // convert radians to degrees

            if (abs(mag - 255) < 1e-4) {
                for (int thetaDegree = oriDegree - delta; thetaDegree < oriDegree + delta; thetaDegree++) {
                    float theta = thetaDegree * (float) M_PI / 180; // converts degrees to radians
                    int rho = (int) (i * sin(theta) + j * cos(theta));
                    if (rho >= 0 && rho < diag && thetaDegree >= 0 && thetaDegree <= 360) {
                        votes[rho][thetaDegree] += 1;
                        hough_space.at<float>(rho,thetaDegree) += 1;
                    }
                }
            }
        }
    }

//    display("hough_space", hough_space);
    vector<Vec2f> lines;
    for (int rho = 0; rho < diag; rho++) {
        for (int thetaDegree = 0; thetaDegree < 361; thetaDegree++) {
            if (votes[rho][thetaDegree] > threshold) {
                double theta = thetaDegree * M_PI / 180;
                lines.insert(lines.end(), Vec2f(rho, (float) theta));
            }
        }
    }

//    imshow("detected lines", src);
//    waitKey(0);

    return lines;
}

/**
 * Colnvolute through image and apply X and Y sobel kernel, then take sqrt
 * @param input
 * @return
 */
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

/**
 * Calculate gradient direction
 *
 * Convolute through image and apply both X and Y sobel, then take atan
 *
 * @param input
 * @return
 */
Mat gradDirection(Mat input) {
    Mat gradXKernel = getXKernel();
    Mat gradYKernel = getYKernel();

    Mat result(input.rows, input.cols, CV_32F);

    int depth = 1;

//    Mat dx(input.rows, input.cols, CV_32F);
//    Mat dy(input.rows, input.cols, CV_32F);

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

//                dx.at<float>(i, j) = sumX;
//                dy.at<float>(i, j) = sumY;
                result.at<float>(i, j) = atan2(sumY, sumX);
            }
        }
    }

//    display("dx", dx);
//    display("dy", dy);
//    normalize(result, result, 0, 1, NORM_MINMAX);
//    imshow("gradient direction", result);
//    waitKey(0);

    return result;
}

/**
 * Apply binary threshold to image
 *
 * @param image
 * @param threshold
 */
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
