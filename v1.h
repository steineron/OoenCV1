//
// Created by Ron Steiner on 14/11/19.
//

#ifndef NC_RD_FACEMASK_V1_H
#define NC_RD_FACEMASK_V1_H

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>

//
//filename := os.Args[1]
///*
//    ratio = image.shape[0] / 500.0
//    orig = image.copy()
//    image = imutils.resize(image, height = 500)
//
//    # convert the image to grayscale, blur it, and find edges
//    # in the image
//    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
//    gray = cv2.GaussianBlur(gray, (5, 5), 0)
//    edged = cv2.Canny(gray, 75, 200)
//
//    # show the original image and the edge detected image
//    print("STEP 1: Edge Detection")
//    cv2.imshow("Image", image)
//    cv2.imshow("Edged", edged)
//    cv2.waitKey(0)
//    cv2.destroyAllWindows()
//*/
//
//mat := gocv.IMRead(filename, gocv.IMReadColor)
//defer mat.Close()
//
//window := displayMat(mat, "grayed")
//window.Close()
//
//
//gocv.Resize(mat, &mat, image.Point{X: 0, Y: 0}, 0.75, 0.75, gocv.InterpolationLinear)
//
//orig := mat.Clone()
//defer orig.Close()
//
//// convert the image to grayscale, blur it, and find edges
//// in the image
//gray := gocv.NewMat()
//defer gray.Close()
//
//gocv.CvtColor(mat, &gray, gocv.ColorBGRToGray)
//
//gocv.GaussianBlur(gray, &gray, image.Point{X: 35, Y: 35}, 0, 0, gocv.BorderDefault)
//gocv.Threshold(gray, &gray, 25, 255, gocv.ThresholdBinaryInv+gocv.ThresholdOtsu)
//// remaining cleanup of the image to use for finding contours.
//// first use threshold
////gocv.Threshold(gray, &gray, 25, 255, gocv.ThresholdBinary)
//
//window = displayMat(gray, "grayed")
//
//edged := gocv.NewMat()
////gocv.Canny(gray, &edged, 75.0, 200.0)
//gocv.Canny(gray, &edged, 150.0, 175.0)
//
//window = displayMat(edged, "edged once")
//
//// then dilate
//kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(6, 6)) // or 5x5?
//defer kernel.Close()
//gocv.Dilate(edged, &edged, kernel)
//window = displayMat(edged, "dilated")
//
///*
//   cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
//   cnts = imutils.grab_contours(cnts)
//   cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
//
//   # loop over the contours
//   for c in cnts:
//       # approximate the contour
//       peri = cv2.arcLength(c, True)
//       approx = cv2.approxPolyDP(c, 0.02 * peri, True)
//
//       # if our approximated contour has four points, then we
//       # can assume that we have found our screen
//       if len(approx) == 4:
//           screenCnt = approx
//           break
//
//   # show the contour (outline) of the piece of paper
//   print("STEP 2: Find contours of paper")
//   cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
//   cv2.imshow("Outline", image)
//   cv2.waitKey(0)
//   cv2.destroyAllWindows()
//*/
//
//contours := gocv.FindContours(edged.Clone(), gocv.RetrievalTree, gocv.ChainApproxSimple)
////contours=contours[0]
////sort.Sort(Contour(contours))
//var screenContour [][]image.Point = make([][]image.Point, 1)
//for _, c := range contours {
//        peri := gocv.ArcLength(c, true)
//        approx := gocv.ApproxPolyDP(c, 0.02*peri, true)
//
//        print("peri:", peri, "\n")
//        print("approx:", approx, "\n")
//
//        //defects := gocv.NewMat()
//        //defer defects.Close()
//        //
//        //gocv.ConvexHull(c, &hull, true, false)
//        //gocv.ConvexityDefects(c, hull, &defects)
//        // if our approximated contour has four points, then we
//        // can assume that we have found our screen
//        if len(approx) == 4 {
//            screenContour[0] = approx
//            break
//        }
//        //screenContour[0] = contours[0]
//
//        //
//        //status := "Motion detected"
//        //statusColor := color.RGBA{0, 255, 0, 0}
//        //gocv.DrawContours(&img, contours, i, statusColor, 2)
//        //
//        //rect := gocv.BoundingRect(c)
//        //gocv.Rectangle(&img, rect, color.RGBA{0, 0, 255, 0}, 2)
//}
//
//for i := 0; i < len(contours); i++ {
//display := mat.Clone()
//gocv.DrawContours(&display, contours, i, color.RGBA{R: 0, G: 255, B: 0, A: 50}, 2)
//window.IMShow(display)
//if window.WaitKey(0) >= 0 {
//continue
//}
////display.Close()//
//}
//
//window.Close()
//
//// detect lines example
//
//mat = edged//orig
//matCanny := gocv.NewMat()
//matLines := gocv.NewMat()
//
//window = gocv.NewWindow("detected lines")
//
//gocv.Canny(mat, &matCanny, 50, 200)
//gocv.HoughLinesP(matCanny, &matLines, 1, math.Pi/180, 40)
//
//fmt.Println(matLines.Cols())
//fmt.Println(matLines.Rows())
//tl := image.Point{X: mat.Size()[0] + 1, Y: mat.Size()[1] + 1}
//br := image.Point{X: -1, Y: -1}
//
//fmt.Println("TL: ", tl)
//fmt.Println("BR: ", br)
//
//for i := 0; i < matLines.Rows(); i++ {
//pt1 := image.Pt(int(matLines.GetVeciAt(i, 0)[0]), int(matLines.GetVeciAt(i, 0)[1]))
//pt2 := image.Pt(int(matLines.GetVeciAt(i, 0)[2]), int(matLines.GetVeciAt(i, 0)[3]))
//gocv.Line(&mat, pt1, pt2, color.RGBA{0, 255, 0, 50}, 2)
//fmt.Println("Pt1:(", pt1.X, ", ", pt1.Y, ")")
//fmt.Println("Pt2: ", pt2)
//
//tl.Y = Min(tl.Y, pt1.Y)
//tl.Y = Min(tl.Y, pt2.Y)
//
//br.Y = Max(br.Y, pt1.Y)
//br.Y = Max(br.Y, pt2.Y)
//
//tl.X = Min(tl.X, pt1.X)
//tl.X = Min(tl.X, pt2.X)
//
//br.X = Max(br.X, pt1.X)
//br.X = Max(br.X, pt2.X)
//
//fmt.Println("TL: ", tl)
//fmt.Println("BR: ", br)
//}
//
//displayMat(mat, "Lines")
//
//snapped := color.RGBA{0, 255, 0, 50}
//crop := image.Rectangle{Min: tl, Max: br}
//gocv.Rectangle(&orig, crop, snapped, 2)
//displayMat(orig, "Result")
//
//
//displayMat(orig.Region(crop), "Cropped")



using namespace cv;
using namespace std;

class ScannerV1 {

public:
    Mat image, gray, blurImage, edge1, edge2, cedge;


};


#endif //NC_RD_FACEMASK_V1_H
