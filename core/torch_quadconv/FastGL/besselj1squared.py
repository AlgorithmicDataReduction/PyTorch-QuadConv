#! /usr/bin/env python
#
def besselj1squared ( k ):

#*****************************************************************************80
#
## BESSELJ1SQUARED computes the square of BesselJ(1, BesselZero(0,k))
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
#
#    Output, real Z, the value of the square of the Bessel
#    J1 function at the K-th zero of the Bessel J0 function.
#
  import numpy as np

  j1 = np.array ( [ \
    0.269514123941916926139021992911E+00, \
    0.115780138582203695807812836182E+00, \
    0.0736863511364082151406476811985E+00, \
    0.0540375731981162820417749182758E+00, \
    0.0426614290172430912655106063495E+00, \
    0.0352421034909961013587473033648E+00, \
    0.0300210701030546726750888157688E+00, \
    0.0261473914953080885904584675399E+00, \
    0.0231591218246913922652676382178E+00, \
    0.0207838291222678576039808057297E+00, \
    0.0188504506693176678161056800214E+00, \
    0.0172461575696650082995240053542E+00, \
    0.0158935181059235978027065594287E+00, \
    0.0147376260964721895895742982592E+00, \
    0.0137384651453871179182880484134E+00, \
    0.0128661817376151328791406637228E+00, \
    0.0120980515486267975471075438497E+00, \
    0.0114164712244916085168627222986E+00, \
    0.0108075927911802040115547286830E+00, \
    0.0102603729262807628110423992790E+00, \
    0.00976589713979105054059846736696E+00 ] )

  if ( 21 < k ):

    x = 1.0E+00 / ( k - 0.25E+00 )
    x2 = x * x
    z = \
        x *       (  0.202642367284675542887758926420E+00 \
      + x2 * x2 * ( -0.303380429711290253026202643516E-03 \
      + x2 *      (  0.198924364245969295201137972743E-03 \
      + x2 *      ( -0.228969902772111653038747229723E-03 \
      + x2 *      (  0.433710719130746277915572905025E-03 \
      + x2 *      ( -0.123632349727175414724737657367E-02 \
      + x2 *      (  0.496101423268883102872271417616E-02 \
      + x2 *      ( -0.266837393702323757700998557826E-01 \
      + x2 *      (  0.185395398206345628711318848386E+00 ) ))))))))
  else:
    z = j1[k-1]

  return z

def besselj1squared_test ( ):

#*****************************************************************************80
#
## BESSELJ1SQUARED_TEST tests BESSELJ1SQUARED.
#
#  Discussion:
#
#    SCIPY.SPECIAL provides the built in function J1.
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
  from besseljzero import besseljzero

  print ( '' )
  print ( 'BESSELJ1SQUARED_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  BESSELJ1SQUARED returns the square of the Bessel J1(X) function' )
  print ( '  at the K-th zero of J0(X).' )
  print ( '' )
  print ( '   K           X(K)                    J1(X(K))^2                 BESSELJ1SQUARED' )
  print ( '' )

  for k in range ( 1, 31 ):
    x = besseljzero ( k )
    f1 = sp.j1 ( x ) ** 2
    f2 = besselj1squared ( k )
    print ( '  %2d  %24.16g  %24.16g  %24.16g' % ( k, x, f1, f2 ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'BESSELJ1SQUARED_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  besselj1squared_test ( )
  timestamp ( )

