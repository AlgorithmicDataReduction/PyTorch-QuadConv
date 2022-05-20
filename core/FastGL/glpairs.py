#! /usr/bin/env python
#
def glpairs ( n, k ):

#*****************************************************************************80
#
## GLPAIRS computes the K-th pair of an N-point Gauss-Legendre rule.
#
#  Discussion:
#
#    This routine is intended for cases were 100 < N.
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
#    1 <= N.
#
#    Input, integer K, the index of the point to be returned.
#    1 <= K <= N.
#
#    Output, real THETA, WEIGHT, X, the theta coordinate, weight, 
#    and x coordinate of the point.
#
  import numpy as np
  from FastGL.besselj1squared import besselj1squared
  from FastGL.besseljzero import besseljzero
  from sys import exit

  if ( n < 1 ):
    print ( '' )
    print ( 'GLPAIRS - Fatal error!' )
    print ( '  Illegal value of N.' )
    exit ( 'GLPAIRS - Fatal error!' )

  if ( k < 1 or n < k ):
    print ( '' )
    print ( 'GLPAIRS - Fatal error!' )
    print ( '  Illegal value of K.' )
    exit ( 'GLPAIRS - Fatal error!' )

  if ( n < ( 2 * k - 1 ) ):
    kcopy = n - k + 1
  else:
    kcopy = k
#
#  Get the Bessel zero.
#
  w = 1.0E+00 / ( float ( n ) + 0.5E+00 )
  nu = besseljzero ( kcopy )
  theta = w * nu
  y = theta ** 2
#
#  Get the asymptotic BesselJ(1,nu) squared.
#
  b = besselj1squared ( kcopy )
#
#  Get the Chebyshev interpolants for the nodes.
#
  sf1t = ((((( \
    - 1.29052996274280508473467968379E-12 * y \
    + 2.40724685864330121825976175184E-10 ) * y \
    - 3.13148654635992041468855740012E-08 ) * y \
    + 0.275573168962061235623801563453E-05 ) * y \
    - 0.148809523713909147898955880165E-03 ) * y \
    + 0.416666666665193394525296923981E-02 ) * y \
    - 0.416666666666662959639712457549E-01

  sf2t = ((((( \
    + 2.20639421781871003734786884322E-09 * y \
    - 7.53036771373769326811030753538E-08 ) * y \
    + 0.161969259453836261731700382098E-05 ) * y \
    - 0.253300326008232025914059965302E-04 ) * y \
    + 0.282116886057560434805998583817E-03 ) * y \
    - 0.209022248387852902722635654229E-02 ) * y \
    + 0.815972221772932265640401128517E-02

  sf3t = ((((( \
    - 2.97058225375526229899781956673E-08 * y \
    + 5.55845330223796209655886325712E-07 ) * y \
    - 0.567797841356833081642185432056E-05 ) * y \
    + 0.418498100329504574443885193835E-04 ) * y \
    - 0.251395293283965914823026348764E-03 ) * y \
    + 0.128654198542845137196151147483E-02 ) * y \
    - 0.416012165620204364833694266818E-02
#
#  Get the Chebyshev interpolants for the weights.
#
  wsf1t = (((((((( \
    - 2.20902861044616638398573427475E-14 * y \
    + 2.30365726860377376873232578871E-12 ) * y \
    - 1.75257700735423807659851042318E-10 ) * y \
    + 1.03756066927916795821098009353E-08 ) * y \
    - 4.63968647553221331251529631098E-07 ) * y \
    + 0.149644593625028648361395938176E-04 ) * y \
    - 0.326278659594412170300449074873E-03 ) * y \
    + 0.436507936507598105249726413120E-02 ) * y \
    - 0.305555555555553028279487898503E-01 ) * y \
    + 0.833333333333333302184063103900E-01

  wsf2t = ((((((( \
    + 3.63117412152654783455929483029E-12 * y \
    + 7.67643545069893130779501844323E-11 ) * y \
    - 7.12912857233642220650643150625E-09 ) * y \
    + 2.11483880685947151466370130277E-07 ) * y \
    - 0.381817918680045468483009307090E-05 ) * y \
    + 0.465969530694968391417927388162E-04 ) * y \
    - 0.407297185611335764191683161117E-03 ) * y \
    + 0.268959435694729660779984493795E-02 ) * y \
    - 0.111111111111214923138249347172E-01

  wsf3t = ((((((( \
    + 2.01826791256703301806643264922E-09 * y \
    - 4.38647122520206649251063212545E-08 ) * y \
    + 5.08898347288671653137451093208E-07 ) * y \
    - 0.397933316519135275712977531366E-05 ) * y \
    + 0.200559326396458326778521795392E-04 ) * y \
    - 0.422888059282921161626339411388E-04 ) * y \
    - 0.105646050254076140548678457002E-03 ) * y \
    - 0.947969308958577323145923317955E-04 ) * y \
    + 0.656966489926484797412985260842E-02
#
#  Refine with the paper expansions.
#
  nuosin = nu / np.sin ( theta )
  bnuosin = b * nuosin
  winvsinc = w * w * nuosin
  wis2 = winvsinc * winvsinc
# 
#  Finally compute the node and the weight.
#
  theta = w * ( nu + theta * winvsinc \
    * ( sf1t + wis2 * ( sf2t + wis2 * sf3t ) ) )
  deno = bnuosin + bnuosin * wis2 * ( wsf1t + wis2 * ( wsf2t + wis2 * wsf3t ) )
  weight = ( 2.0E+00 * w ) / deno

  if ( n < ( 2 * k - 1 ) ):
    theta = np.pi - theta

  x = np.cos ( theta )

  return theta, weight, x

def glpairs_test ( ):

#*****************************************************************************80
#
## GLPAIRS_TEST tests GLPAIRS.
#
#  Discussion:
#
#    Test the numerical integration of cos(1000 x) over the range [-1,1]
#    for varying number of Gauss-Legendre quadrature nodes l.
#    The fact that only twelve digits of accuracy are obtained is due to the 
#    condition number of the summation.
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
  print ( 'GLPAIRS_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  integral ( -1 <= x <= 1 ) cos(1000 x) dx' )
  print ( '' )
  print ( '    Nodes           Estimate' )
  print ( '' )

  for l in range ( 500, 620, 20 ):

    q = 0.0

    for k in range ( 1, l + 1 ):
      theta, weight, x = glpairs ( l, k )
      q = q + weight * np.cos ( 1000.0 * x )

    print ( '  %7d  %24.16g' % ( l, q ) )

  print ( '' )
  print ( '    Exact  %24.16g' % ( 0.002 * np.sin ( 1000.0 ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'GLPAIRS_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  glpairs_test ( )
  timestamp ( )

