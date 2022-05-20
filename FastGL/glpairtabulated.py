#! /usr/bin/env python
#
def glpairtabulated ( l, k ):

#*****************************************************************************80
#
## GLPAIRTABULATED computes the K-th pair of an N-point Gauss-Legendre rule.
#
#  Discussion:
#
#    Data is tabulated for 1 <= L <= 100.
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
#    Input, integer L, the number of points in the given rule.
#    1 <= L <= 100.
#
#    Input, integer K, the index of the point to be returned.
#    1 <= K <= L.
#
#    Output, real THETA, WEIGHT, X, the theta coordinate, weight, 
#    and x coordinate of the point.
#
  import numpy as np
  from FastGL.legendre_theta import legendre_theta
  from FastGL.legendre_weight import legendre_weight
  from sys import exit

  if ( l < 1 or 100 < l ):
    print ( '' )
    print ( 'GLPAIRTABULATED - Fatal error!' )
    print ( '  Illegal value of L.' )
    exit ( 'GLPAIRTABULATED - Fatal error!' )

  if ( k < 1 or l < k ):
    print ( '' )
    print ( 'GLPAIRTABULATED - Fatal error!' )
    print ( '  Illegal value of K.' )
    exit ( 'GLPAIRTABULATED - Fatal error!' )

  theta = legendre_theta ( l, k )
  weight = legendre_weight ( l, k )

  x = np.cos ( theta )   

  return theta, weight, x

def glpairtabulated_test ( ):

#*****************************************************************************80
#
## GLPAIRTABULATED_TEST tests GLPAIRTABULATED.
#
#  Discussion:
#
#    Test the numerical integration of exp(x) over the range [-1,1]
#    for varying number of Gauss-Legendre quadrature nodes l.
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
  print ( 'GLPAIRTABULATED_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  integral ( -1 <= x <= 1 ) exp(x) dx' )
  print ( '' )
  print ( '    Nodes           Estimate' )
  print ( '' )

  for l in range ( 1, 10 ):
    q = 0.0
    for k in range ( 1, l + 1 ):
      theta, weight, x = glpairtabulated ( l, k )
      q = q + weight * np.exp ( x )
    print ( '  %7d  %24.16g' % ( l, q ) )

  print ( '' )
  print ( '    Exact  %24.16g' % ( np.exp ( 1.0E+00 ) - np.exp ( -1.0E+00 ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'GLPAIRTABULATED_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  glpairtabulated_test ( )
  timestamp ( )


