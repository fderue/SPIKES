 /*
 @author Derue François-Xavier
 francois.xavier.derue@gmail.ca

 Some utily functions for Slic_Cuda
 */
#pragma once


/* determine width and height (integer value) of an initial spx from its expected size d. */
void getWlHl(int w, int h, int d, int & wl, int & hl);
inline int iDivUp(int a, int b){ return (a%b==0)? a/b : a/b+1; }