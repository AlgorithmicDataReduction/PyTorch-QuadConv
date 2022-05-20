#! /usr/bin/env python
#
def fastgl_test ( ):

#*****************************************************************************80
#
## FASTGL_TEST tests the FASTGL library.
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
  import platform
  from besseljzero      import besseljzero_test
  from besselj1squared  import besselj1squared_test
  from glpair           import glpair_test
  from glpairs          import glpairs_test
  from glpairtabulated  import glpairtabulated_test
  from legendre_theta   import legendre_theta_test
  from legendre_weight  import legendre_weight_test

  print ( '' )
  print ( 'FASTGL_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Test the FASTGL library.' )

  besseljzero_test ( )
  besselj1squared_test ( )
  glpair_test ( )
  glpairs_test ( )
  glpairtabulated_test ( )
  legendre_theta_test ( )
  legendre_weight_test ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'FASTGL_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  fastgl_test ( )
  timestamp ( )

