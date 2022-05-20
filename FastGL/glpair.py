#! /usr/bin/env python
#
def glpair ( n, k ):

#*****************************************************************************80
#
## GLPAIR computes the K-th pair of an N-point Gauss-Legendre rule.
#
#  Discussion:
#
#    If N <= 100, GLPAIRTABULATED is called, otherwise GLPAIR is called.
#
#    Theta values of the zeros are in [0,pi], and monotonically increasing. 
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 January 2016
#
#  Author:
#
#    Original C++ version by Ignace Bogaert.
#    Python version by John Burkardt.
#
#  Reference:
#
#    Ignace Bogaert,
#    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
#    SIAM Journal on Scientific Computing,
#    Volume 36, Number 3, 2014, pages A1008-1026.
#
#  Parameters:
#
#    Input, integer N, the number of points in the given rule.
#    0 < N.
#
#    Input, integer K, the index of the point to be returned.
#    1 <= K <= N.
#
#    Output, real THETA, WEIGHT, X, the theta coordinate, weight, 
#    and x coordinate of the point.
#
  from FastGL.glpairs import glpairs
  from FastGL.glpairtabulated import glpairtabulated
  from sys import exit

  if ( n < 1 ):
    print ( '' )
    print ( 'GLPAIR - Fatal error!' )
    print ( '  Illegal value of N.' )
    exit ( 'GLPAIR - Fatal error!' )

  if ( k < 1 or n < k ):
    print ( '' )
    print ( 'GLPAIR - Fatal error!' )
    print ( '  Illegal value of K.' )
    exit ( 'GLPAIR - Fatal error!' )

  if ( n < 101 ):
    theta, weight, x = glpairtabulated ( n, k )
  else:
    theta, weight, x = glpairs ( n, k )

  return theta, weight, x

def glpair_test ( ):

#*****************************************************************************80
#
## GLPAIR_TEST tests GLPAIR.
#
#  Discussion:
#
#    Test the numerical integration of ln(x) over the range [0,1]
#    Normally, one would not use Gauss-Legendre quadrature for this,
#    but for the sake of having an example with l > 100, this is included.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 January 2016
#
#  Author:
#
#    Original C++ version by Ignace Bogaert.
#    Python version by John Burkardt.
#
#  Reference:
#
#    Ignace Bogaert,
#    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
#    SIAM Journal on Scientific Computing,
#    Volume 36, Number 3, 2014, pages A1008-1026.
#
  import numpy as np
  import platform

  print ( '' )
  print ( 'GLPAIR_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Estimate integral ( 0 <= x <= 1 ) ln(x) dx.' )
  print ( '' )
  print ( '    Nodes           Estimate' )
  print ( '' )

  l = 1
  for p in range ( 0, 7 ):
    q = 0.0
    for k in range ( 1, l + 1 ):
      theta, weight, x = glpair ( l, k )
      q = q + 0.5 * weight * np.log ( 0.5 * ( x + 1.0 ) )
    print ( '  %7d       %24.16g' % ( l, q ) )
    l = l * 10
  print ( '' )
  print ( '    Exact        -1.0' )
#
#  Terminate.
#
  print ( '' )
  print ( 'GLPAIR_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  glpair_test ( )
  timestamp ( )


