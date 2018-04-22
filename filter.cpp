#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "filter.hpp"

using namespace std;
using namespace cv;

/**
 *
 */
Filter::Filter(){}


/**
 *
 */
Filter::~Filter(){}


/**
 * Transform image in grayscale, method simple, sum three channels and divide by three.
 *
 * @param Mat &imgIn - Input image.
 * @param Mat &imgOut - Output image.
 */
void Filter::grayscaleSimple(Mat &imgIn, Mat &imgOut){  
  int gray;
  int height = imgIn.size().height;
  int width = imgIn.size().width;
  
  imgOut = Mat::zeros(height, width, CV_8UC3);

  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      gray = ( imgIn.at<Vec3b>(y,x)[0] + imgIn.at<Vec3b>(y,x)[1] + imgIn.at<Vec3b>(y,x)[2] )/3;
    
      imgOut.at<Vec3b>(y,x)[0] = gray;
      imgOut.at<Vec3b>(y,x)[1] = gray; 
      imgOut.at<Vec3b>(y,x)[2] = gray; 
    }
  }
}


/**
 * Transform image in grayscale, method pondered, multiply the channels by the weights.
 *    [CHANNEL RED] * 0.299
 *    [CHANNEL GREEN] * 0.587
 *    [CHANNEL BLUE] * 0.114
 *
 * @param Mat &imgIn - Input image.
 * @param Mat &imgOut - Output image.
 */
void Filter::grayscale(Mat &imgIn, Mat &imgOut){
  int gray;  
  int height = imgIn.size().height;
  int width = imgIn.size().width;
  
  imgOut = Mat::zeros(height, width, CV_8UC3);

  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      gray = ( 0.114*imgIn.at<Vec3b>(y,x)[0] + 0.587*imgIn.at<Vec3b>(y,x)[1] + 0.299*imgIn.at<Vec3b>(y,x)[2] );

      imgOut.at<Vec3b>(y,x)[0] = gray;
      imgOut.at<Vec3b>(y,x)[1] = gray; 
      imgOut.at<Vec3b>(y,x)[2] = gray; 
    }
  }
}


/**
 * Apply zoom out in image. 
 *
 * @param Mat &imgIn - Input image.
 * @param Mat &imgOut - Output image.
 */
void Filter::zoomOut(Mat &imgIn, Mat &imgOut){
  Vec3f pixelOut;
  int height = imgIn.size().height;
  int width = imgIn.size().width;

  height += ((height % 2) == 0) ? 0 : 1;
  width += ((width % 2) == 0) ? 0 : 1;

  imgOut = Mat::zeros(height/2, width/2, CV_8UC3);

  for(int y = 0; y < height; y+=2){
    for(int x = 0; x < width; x+=2){        
      pixelOut.val[0] = ( imgIn.at<Vec3b>(y,x)[0] + imgIn.at<Vec3b>(y+1,x)[0] + imgIn.at<Vec3b>(y,x+1)[0] + imgIn.at<Vec3b>(y+1,x+1)[0] )/4;
      pixelOut.val[1] = ( imgIn.at<Vec3b>(y,x)[1] + imgIn.at<Vec3b>(y+1,x)[1] + imgIn.at<Vec3b>(y,x+1)[1] + imgIn.at<Vec3b>(y+1,x+1)[1] )/4;
      pixelOut.val[2] = ( imgIn.at<Vec3b>(y,x)[2] + imgIn.at<Vec3b>(y+1,x)[2] + imgIn.at<Vec3b>(y,x+1)[2] + imgIn.at<Vec3b>(y+1,x+1)[2] )/4;

      imgOut.at<Vec3b>(y/2,x/2) = pixelOut;
    }
  }    
}


/**
 * Apply zoom in image.
 *
 * @param Mat &imgIn - Input image.
 * @param Mat &imgOut - Output image.
 */
void Filter::zoomIn(Mat &imgIn, Mat &imgOut){
  int height = imgIn.size().height;
  int width = imgIn.size().width;

  imgOut = Mat::zeros(height*2, width*2, CV_8UC3);

  for(int y = 0; y < height*2; y++){
    for(int x = 0; x < width*2; x++){
      imgOut.at<Vec3b>(y,x) = imgIn.at<Vec3b>((int) y/2, (int) x/2);
    }
  }
}


/**
 * Apply filter thresholding in image, based on '@param limit'.
 *
 * @param Mat &imgIn - Input image.
 * @param int limit - Limit for apply in image.
 * @param Mat &imgOut - Output image.
 */
void Filter::thresholding(Mat &imgIn, int limit, Mat &imgOut){
  int height = imgIn.size().height;
  int width = imgIn.size().width;
  int pixel;

  imgOut = Mat::zeros(height, width, CV_8UC3);

  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      pixel = (( imgIn.at<Vec3b>(y,x)[0] + imgIn.at<Vec3b>(y,x)[1] + imgIn.at<Vec3b>(y,x)[2] ) > limit) ? 255 : 0;
      imgOut.at<Vec3b>(y,x)[0] = pixel;
      imgOut.at<Vec3b>(y,x)[1] = pixel;
      imgOut.at<Vec3b>(y,x)[2] = pixel;
    }
  }
}


/**
 * Apply filter negative in image.
 *
 * @param Mat &imgIn - Input image.
 * @param Mat &imgOut - Output image.
 */
void Filter::negative(Mat &imgIn, Mat &imgOut){
  int height = imgIn.size().height;
  int width = imgIn.size().width;

  imgOut = Mat::zeros(height, width, CV_8UC3);

  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      imgOut.at<Vec3b>(y,x)[0] = 255 - imgIn.at<Vec3b>(y,x)[0];
      imgOut.at<Vec3b>(y,x)[1] = 255 - imgIn.at<Vec3b>(y,x)[1];
      imgOut.at<Vec3b>(y,x)[2] = 255 - imgIn.at<Vec3b>(y,x)[2];
    }
  }
}


/**
 * Transform two image in one, adding both.
 *
 * @param Mat &imgInA - Input image A.
 * @param Mat &imgInB - Input image B.
 * @param Mat &imgOut - Output image.
 * @param int weightA - Weight to apply in image A.
 * @param int weightB - Weight to apply in image B.
 */
void Filter::addition(Mat &imgInA, Mat &imgInB, Mat &imgOut, int weightA, int weightB){
  int height = ( imgInA.size().height <= imgInB.size().height ) ? imgInA.size().height : imgInB.size().height;
  int width = ( imgInA.size().width <= imgInB.size().width ) ? imgInA.size().width : imgInB.size().width;

  imgOut = Mat::zeros(height, width, CV_8UC3);

  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      imgOut.at<Vec3b>(y,x)[0] = (weightA * imgInA.at<Vec3b>(y,x)[0] + weightB * imgInB.at<Vec3b>(y,x)[0])/(weightA + weightB);
      imgOut.at<Vec3b>(y,x)[1] = (weightA * imgInA.at<Vec3b>(y,x)[1] + weightB * imgInB.at<Vec3b>(y,x)[1])/(weightA + weightB);
      imgOut.at<Vec3b>(y,x)[2] = (weightA * imgInA.at<Vec3b>(y,x)[2] + weightB * imgInB.at<Vec3b>(y,x)[2])/(weightA + weightB);
    }
  }
}


/**
 *
 */
void Filter::subtraction(Mat &imgInA, Mat &imgInB, Mat &imgOut){
  int height = ( imgInA.size().height <= imgInB.size().height ) ? imgInA.size().height : imgInB.size().height;
  int width = ( imgInA.size().width <= imgInB.size().width ) ? imgInA.size().width : imgInB.size().width;

  imgOut = Mat::zeros(height, width, CV_8UC3);

  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      imgOut.at<Vec3b>(y,x)[0] = validateRange(imgInA.at<Vec3b>(y,x)[0] - imgInB.at<Vec3b>(y,x)[0]);
      imgOut.at<Vec3b>(y,x)[1] = validateRange(imgInA.at<Vec3b>(y,x)[1] - imgInB.at<Vec3b>(y,x)[1]);
      imgOut.at<Vec3b>(y,x)[2] = validateRange(imgInA.at<Vec3b>(y,x)[2] - imgInB.at<Vec3b>(y,x)[2]);
    }
  }
}


/**
 *
 */
void Filter::isolateChannels(Mat &imgIn, Mat &imgOut, bool red, bool green, bool blue){
  int height = imgIn.size().height;
  int width = imgIn.size().width;

  imgOut = Mat::zeros(height, width, CV_8UC3);

  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      imgOut.at<Vec3b>(y,x)[0] = (blue) ? imgIn.at<Vec3b>(y,x)[0] : 0;
      imgOut.at<Vec3b>(y,x)[1] = (green) ? imgIn.at<Vec3b>(y,x)[1] : 0;
      imgOut.at<Vec3b>(y,x)[2] = (red) ? imgIn.at<Vec3b>(y,x)[2] : 0; 
    }
  }
}


/**
 *
 */
void Filter::incrementChannels(Mat &imgIn, Mat &imgOut, int red, int green, int blue){
  int height = imgIn.size().height;
  int width = imgIn.size().width;

  imgOut = Mat::zeros(height, width, CV_8UC3);

  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      imgOut.at<Vec3b>(y,x)[0] = validateRange( imgIn.at<Vec3b>(y,x)[0] + blue );
      imgOut.at<Vec3b>(y,x)[1] = validateRange( imgIn.at<Vec3b>(y,x)[1] + green );
      imgOut.at<Vec3b>(y,x)[2] = validateRange( imgIn.at<Vec3b>(y,x)[2] + red );
    }
  }
}


/**
 *
 */
void Filter::histogram(Mat &imgIn, Mat &imgOut){
  /* Separate the image in 3 places ( B, G and R ) */
  vector<Mat> bgr_planes;
  split( imgIn, bgr_planes );

  /* Establish the number of bins */
  int histSize = 256;

  /* Set the ranges ( for B,G,R) ) */
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat b_hist, g_hist, r_hist;

  /* Compute the histograms: */
  calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

  /* Draw the histograms for B, G and R */
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /* Normalize the result to [ 0, histImage.rows ] */
  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /* Draw for each channel */
  for( int i = 1; i < histSize; i++ ){
    line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ), Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ), Scalar( 255, 0, 0), 2, 8, 0  );
    line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ), Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ), Scalar( 0, 255, 0), 2, 8, 0  );
    line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ), Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ), Scalar( 0, 0, 255), 2, 8, 0  );
  }

  /* Display */
  namedWindow("Histograma", CV_WINDOW_AUTOSIZE );
  imshow("Histograma", histImage );
}

/**
 *
 */
/* void Filter:: outgoingPoints(Mat &imgIn, Mat &imgOut, int mask[][], int n) */
void Filter::outgoingPoints(Mat &imgIn, Mat &imgOut){ 
  int pixel;
  int mask[3][3] = {{-1, -1, -1},{-1, 8, -1},{-1, -1, -1}};
  int n = sizeof(mask)/sizeof(mask[0]);
  int border = n/2;
  int height = imgIn.size().height;
  int width = imgIn.size().width;

  imgOut = Mat::zeros(height, width, CV_8UC3);

  for(int y = border; y < height-border; y++){
    for(int x = border; x < width-border; x++){
      pixel = 0;

      for(int j = 0-border; j <= 0+border; j++){
        for(int i = 0-border; i <= 0+border; i++){
          pixel += (imgIn.at<Vec3b>(y+j,x+i)[0] * mask[j+border][i+border]); 
        }
      }
      imgOut.at<Vec3b>(y,x)[0] = validateRange(pixel);
      imgOut.at<Vec3b>(y,x)[1] = validateRange(pixel);
      imgOut.at<Vec3b>(y,x)[2] = validateRange(pixel);
    }
  }
}

/**
 *
 */
int Filter::validateRange(int channel){
  return (channel > 255) ? 255 : (channel < 0) ? 0 : channel;
}


































