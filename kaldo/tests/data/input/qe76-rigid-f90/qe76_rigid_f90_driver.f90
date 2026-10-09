! SPDX-License-Identifier: GPL-2.0-only
PROGRAM qe76_rigid_f90_driver
  USE kinds, ONLY : DP
  USE rigid, ONLY : rgd_blk, nonanal
  IMPLICIT NONE
  INTEGER :: nr1,nr2,nr3,nat,iu,ios,i,j,na,nb,itau(3)
  REAL(DP) :: alat,omega,alph,at(3,3),bg(3,3),tau(3,3),epsil(3,3),zeu(3,3,3)
  REAL(DP) :: q_finite_3d(3),q_gamma_direction(3),q_finite_2d(3),q(3)
  COMPLEX(DP) :: dyn(3,3,3,3)
  CHARACTER(LEN=256) :: input_file,output_file
  NAMELIST /rigid_fixture/ nr1,nr2,nr3,nat,alat,omega,alph,at,bg,tau,epsil,zeu, &
       q_finite_3d,q_gamma_direction,q_finite_2d
  CALL GET_COMMAND_ARGUMENT(1,input_file)
  CALL GET_COMMAND_ARGUMENT(2,output_file)
  IF (LEN_TRIM(input_file)==0 .OR. LEN_TRIM(output_file)==0) ERROR STOP 'usage: driver input.nml output.dat'
  OPEN(NEWUNIT=iu,FILE=TRIM(input_file),STATUS='old',ACTION='read',IOSTAT=ios)
  IF (ios/=0) ERROR STOP 'cannot open input'; READ(iu,NML=rigid_fixture,IOSTAT=ios); CLOSE(iu)
  IF (ios/=0 .OR. nat/=3) ERROR STOP 'bad fixture input'
  itau=[1,2,3]; OPEN(NEWUNIT=iu,FILE=TRIM(output_file),STATUS='replace',ACTION='write')
  q=MATMUL(bg,q_finite_3d); dyn=(0.0_DP,0.0_DP)
  CALL rgd_blk(nr1,nr2,nr3,nat,dyn,q,tau,epsil,zeu,alph,bg,omega,alat,.FALSE.,1.0_DP)
  CALL emit(iu,'finite_3d',dyn)
  q=0.0_DP; dyn=(0.0_DP,0.0_DP)
  CALL rgd_blk(nr1,nr2,nr3,nat,dyn,q,tau,epsil,zeu,alph,bg,omega,alat,.FALSE.,1.0_DP)
  CALL emit(iu,'gamma_direct_3d',dyn)
  CALL nonanal(nat,nat,itau,epsil,MATMUL(bg,q_gamma_direction),zeu,omega,dyn)
  CALL emit(iu,'gamma_directional_3d',dyn)
  q=MATMUL(bg,q_finite_3d); dyn=(0.0_DP,0.0_DP)
  CALL rgd_blk(nr1,nr2,nr3,nat,dyn,q,tau,epsil,zeu,alph,bg,omega,alat,.FALSE.,-1.0_DP)
  CALL emit(iu,'finite_negative_3d',dyn)
  q=MATMUL(bg,q_finite_2d); dyn=(0.0_DP,0.0_DP)
  CALL rgd_blk(nr1,nr2,nr3,nat,dyn,q,tau,epsil,zeu,alph,bg,omega,alat,.TRUE.,1.0_DP)
  CALL emit(iu,'finite_2d',dyn); CLOSE(iu)
CONTAINS
  SUBROUTINE emit(unit,label,value)
    INTEGER,INTENT(IN)::unit
    CHARACTER(*),INTENT(IN)::label
    COMPLEX(DP),INTENT(IN)::value(3,3,3,3)
    WRITE(unit,'(A)') '['//TRIM(label)//']'
    DO nb=1,3; DO na=1,3; DO j=1,3; DO i=1,3
      WRITE(unit,'(4(I0,1X),2(ES26.17E3,1X))') i,j,na,nb,REAL(value(i,j,na,nb),DP),AIMAG(value(i,j,na,nb))
    END DO; END DO; END DO; END DO
  END SUBROUTINE emit
END PROGRAM qe76_rigid_f90_driver
