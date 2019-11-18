#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>

#include "v1.h"

using namespace cv;
using namespace std;
int edgeThresh = 1;
int edgeThreshScharr = 1;
Mat image, gray, blurImage, edge1, edge2, cedge;
const char *window_name1 = "Edge map : Canny default (Sobel gradient)";
const char *window_name2 = "Edge map : Canny with custom gradient (Scharr)";


const int w = 500;
int levels = 3;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;


// define a trackbar callback
static void processImage(Mat &image) {

    resize(image, image, Size(0, 0), 0.75, 0.75, INTER_AREA);

    // create a BW image:
    cvtColor(image, gray, COLOR_BGR2GRAY);


    // blur it to educe noise
    blur(gray, blurImage, Size(3, 3));

    namedWindow("temp", 1);
    imshow("temp", blurImage);
    waitKey(0);


    Mat binary;
    binary.create(gray.size(), CV_8UC1);
    // convert to binary image
    threshold(blurImage, binary, 25, 255, THRESH_BINARY_INV + THRESH_OTSU);

    imshow("temp", binary);
    waitKey(0);

    if (binary.type() != CV_8UC1) {
        binary.convertTo(binary, CV_8UC1);
        imshow("temp", binary);
        waitKey(0);

    }
    // dilate, erode, dilate to further remove noise and small objects
    int niters = 3; // in 3 iterations

    dilate(binary, binary, Mat(), Point(-1, -1), niters);
    erode(binary, binary, Mat(), Point(-1, -1), niters * 2);
    dilate(binary, binary, Mat(), Point(-1, -1), niters);

    // Run the edge detector on grayscale
    Canny(binary, edge1, edgeThresh, edgeThresh * 3, 3);

    imshow("temp", edge1);
    waitKey(0);

    vector<vector<Point> > contours0;
    findContours(edge1, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    contours.resize(contours0.size());
    for (size_t k = 0; k < contours0.size(); k++)
        approxPolyDP(Mat(contours0[k]), contours[k], 3, true);

    // see example in https://docs.opencv.org/master/db/d00/samples_2cpp_2squares_8cpp-example.html#a17
    // for extracting squares

    // or do this: find the contour with largest area - assume it is the document
    Mat contouredImage = Mat::zeros(w, w, CV_8UC3);
    image.copyTo(contouredImage);
    int _levels = levels - 3;
    int maxArea = 0, max = 0;
    vector<Point> approxPoly;
    for (int i = 0; i < contours.size(); i++) {
        int current = contourArea(contours[i], false);
        if (current > max) {
            // check the shape of it:
            vector<Point> approx;
            approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);

            // square contours should have 4 vertices after approximation
            // relatively large area (to filter out noisy contours)
            // and be convex.
            // Note: absolute value of an area is used because
            if( approx.size() == 4 && isContourConvex(approx) )            {
                max = current;
                maxArea = i;
                approxPoly = approx; // keep record of hte approximated polygon
            }
        }
    }

    drawContours(contouredImage, contours, maxArea, Scalar(128, 255, 255),
                 3, LINE_AA, hierarchy, std::abs(_levels));


    RotatedRect rect = minAreaRect( contours[maxArea]);
    Point2f box[4];
    rect.points(box);
    Scalar blue = Scalar(255, 128, 0);
    Scalar red = Scalar(0, 128, 255);
    for( int j = 0; j < 4; j++ ){
        line(contouredImage, approxPoly[j], approxPoly[(j+1)%4], blue, 3, LINE_AA);
        line(contouredImage, box[j], box[(j+1)%4], red, 3, LINE_AA);
    }

    namedWindow("contours", 1);
    imshow("contours", contouredImage);

}

static void help() {
    printf("\nThis sample demonstrates Canny edge detection\n"
           "Call:\n"
           "    /.edge [image_name -- Default is fruits.jpg]\n\n");
}

const char *keys =
        {
                "{help h||}{@image |fruits.jpg|input image name}"
        };

int main(int argc, const char **argv) {
    help();
    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);
    image = imread(samples::findFile(filename), IMREAD_COLOR);
    if (image.empty()) {
        printf("Cannot read image file: %s\n", filename.c_str());
        help();
        return -1;
    }
    processImage(image);
    // Wait for a key stroke; the same function arranges events processing
    waitKey(0);
    return 0;
}