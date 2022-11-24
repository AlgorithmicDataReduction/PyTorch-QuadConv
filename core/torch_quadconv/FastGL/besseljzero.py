#! /usr/bin/env python
#
def besseljzero ( k ):

#*****************************************************************************80
#
## BESSELJZERO computes the kth zero of the J0(X) Bessel function.
#
#  Discussion:
#
#    Note that the first 20 zeros are tabulated.  After that, they are
#    computed.
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
#    Input, integer K, the index of the desired zero.
#    1 <= K.
#
#    Output, real X, the value of the zero.
#
  import numpy as np

  jz = np.array ( [ \
    2.40482555769577276862163187933E+00, \
    5.52007811028631064959660411281E+00, \
    8.65372791291101221695419871266E+00, \
    11.7915344390142816137430449119E+00, \
    14.9309177084877859477625939974E+00, \
    18.0710639679109225431478829756E+00, \
    21.2116366298792589590783933505E+00, \
    24.3524715307493027370579447632E+00, \
    27.4934791320402547958772882346E+00, \
    30.6346064684319751175495789269E+00, \
    33.7758202135735686842385463467E+00, \
    36.9170983536640439797694930633E+00, \
    40.0584257646282392947993073740E+00, \
    43.1997917131767303575240727287E+00, \
    46.3411883716618140186857888791E+00, \
    49.4826098973978171736027615332E+00, \
    52.6240518411149960292512853804E+00, \
    55.7655107550199793116834927735E+00, \
    58.9069839260809421328344066346E+00, \
    62.0484691902271698828525002646E+00 ] )

  if ( 20 < k ):
    x = np.pi * ( k - 0.25E+00 )
    r = 1.0E+00 / x
    r2 = r * r
    x = x \
      + r  * (  0.125E+00 \
      + r2 * ( -0.807291666666666666666666666667E-01 \
      + r2 * (  0.246028645833333333333333333333E+00 \
      + r2 * ( -0.182443876720610119047619047619E+01 \
      + r2 * (  0.253364147973439050099206349206E+02 \
      + r2 * ( -0.567644412135183381139802038240E+03 \
      + r2 * (  0.186904765282320653831636345064E+05 \
      + r2 * ( -0.849353580299148769921876983660E+06 \
      + r2 *    0.509225462402226769498681286758E+08 ))))))))
  else:
    x = jz[k-1]

  return x

def besseljzero_test ( ):

#*****************************************************************************80
#
## BESSELJZERO_TEST tests BESSELJZERO.
#
#  Discussion:
#
#    SCIPY.SPECIAL provides the built in J0(X) function.
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
#    John Burkardt
#
#  Reference:
#
#    Ignace Bogaert,
#    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
#    SIAM Journal on Scientific Computing,
#    Volume 36, Number 3, 2014, pages A1008-1026.
#
  import platform
  import scipy.special as sp

  print ( '' )
  print ( 'BESSELJZERO_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  BESSELJZERO returns the K-th zero of J0(X).' )
  print ( '' )
  print ( '   K           X(K)                  J0(X(K))' )
  print ( '' )

  for k in range ( 1, 31 ):
    x = besseljzero ( k )
    j0x = sp.j0 ( x )
    print ( '  %2d  %24.16g  %24.16g' % ( k, x, j0x ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'BESSELJZERO_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  besseljzero_test ( )
  timestamp ( )

