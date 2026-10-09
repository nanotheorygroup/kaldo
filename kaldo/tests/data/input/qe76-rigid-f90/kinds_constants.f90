! SPDX-License-Identifier: GPL-2.0-only
! Minimal QE-compatible module interfaces needed to compile rigid.f90.
MODULE kinds
  IMPLICIT NONE
  INTEGER, PARAMETER :: DP = KIND(1.0D0)
END MODULE kinds

MODULE constants
  USE kinds, ONLY : DP
  IMPLICIT NONE
  REAL(DP), PARAMETER :: pi = 3.14159265358979323846_DP
  REAL(DP), PARAMETER :: tpi = 2.0_DP * pi
  REAL(DP), PARAMETER :: fpi = 4.0_DP * pi
  REAL(DP), PARAMETER :: e2 = 2.0_DP
  ! Required to type-check the unused dyndiag procedure in rigid.f90.
  REAL(DP), PARAMETER :: amu_ry = 911.444242132_Dp
END MODULE constants

SUBROUTINE errore(routine,message,code)
  CHARACTER(*), INTENT(IN) :: routine,message
  INTEGER, INTENT(IN) :: code
  ERROR STOP code
END SUBROUTINE errore
