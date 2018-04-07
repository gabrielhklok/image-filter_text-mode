/**
 *
 */
#ifndef FILTER
#define	FILTER

#include <iostream>
#include <string>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

class Filter {    
  public:
    Filter();
    ~Filter();

    void grayscaleSimple(Mat &imgIn, Mat &imgOut);
    void grayscale(Mat &imgIn, Mat &imgOut);
    void zoomIn(Mat &imgIn, Mat &imgOut);
    void zoomOut(Mat &imgIn, Mat &imgOut);
    void thresholding(Mat &imgIn, int limit, Mat &imgOut);
    void negative(Mat &imgIn, Mat &imgOut);
    void addition(Mat &imgInA, Mat &imgInB, Mat &imgOut, int weightA = 1, int weightB = 1);
    void subtraction(Mat &imgInA, Mat &imgInB, Mat &imgOut);
    void isolateChannels(Mat &imgIn, Mat &imgOut, bool red = false, bool green = false, bool blue = false);
    void incrementChannels(Mat &imgIn, Mat &imgOut, int red = 0, int green = 0, int blue = 0);
    void histogram(Mat &imgIn, Mat &imgOut);
   // void outgoingPoints(Mat &imgIn, Mat &imgOut, int mask[][], int n);
    void outgoingPoints(Mat &imgIn, Mat &imgOut);

  private:
    string msgResponseDefault;
    
    int validateRange(int channel);
};

#endif	/* FILTER */
