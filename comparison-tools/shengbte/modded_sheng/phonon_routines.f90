!  ShengBTE, a solver for the Boltzmann Transport Equation for phonons
!  Copyright (C) 2012-2023 Wu Li <wu.li.phys2011@gmail.com>
!  Copyright (C) 2012-2023 Jesús Carrete Montaña <jcarrete@gmail.com>
!  Copyright (C) 2012-2023 Nebil Ayape Katcho <nebil.ayapekatcho@cea.fr>
!  Copyright (C) 2012-2023 Natalio Mingo Bisquert <natalio.mingo@cea.fr>
!  Copyright (C) 2021-2022 Fanchen Meng <fanchem@g.clemson.edu>
!  Copyright (C) 2022-2023 Ben Durham <bd740@york.ac.uk>
!  Copyright (C) 2023 Martí Raya Moreno <marti_94@hotmail.com>
!
!  This program is free software: you can redistribute it and/or modify
!  it under the terms of the GNU General Public License as published by
!  the Free Software Foundation, either version 3 of the License, or
!  (at your option) any later version.
!
!  This program is distributed in the hope that it will be useful,
!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!  GNU General Public License for more details.
!
!  You should have received a copy of the GNU General Public License
!  along with this program.  If not, see <http://www.gnu.org/licenses/>.

! Routines used to calculate the phonon spectrum.
module phonon_routines
  use misc
  use data
  use config
  use input
  use mpi
  implicit none

contains

  ! Create the q-point grid and compute all relevant properties.
  subroutine eigenDM(omega,eigenvect,velocity)
    implicit none

    real(kind=dp),intent(out) :: omega(nbands,nptk),velocity(3,nbands,nptk)
    complex(kind=dp),intent(out) :: eigenvect(Nbands,Nbands,nptk)

    real(kind=dp) :: kspace(nptk,3)
    integer :: indexK,ii,jj,kk
    character(len=1) :: aux

    do ii=1,Ngrid(1)        ! rlattvec(:,1) direction
       do jj=1,Ngrid(2)     ! rlattvec(:,2) direction
          do kk=1,Ngrid(3)  ! rlattvec(:,3) direction
             indexK=((kk-1)*Ngrid(2)+(jj-1))*Ngrid(1)+ii
             kspace(indexK,:)=rlattvec(:,1)*(ii-1.0_dp)/real(ngrid(1),kind=dp)+&
                  rlattvec(:,2)*(jj-1.0_dp)/real(ngrid(2),kind=dp)+&
                  rlattvec(:,3)*(kk-1.0_dp)/real(ngrid(3),kind=dp)
          end do
       end do
    end do
    omega=0.0_dp
    velocity=0.0_dp
    eigenvect=cmplx(0.0_dp,0.0_dp,kind=dp)
    kk=ceiling(real(nptk,kind=dp)/real(numprocs,kind=dp))
    ii=min(nptk,kk*myid)+1
    jj=min(nptk,kk*(myid+1))
    ! The routine to be called depends on the input format, selected through a
    ! flag in the CONTROL file.
    if(espresso) then
       call phonon_espresso(kspace(ii:jj,:),omega(:,ii:jj),&
            velocity(:,:,ii:jj),eigenvect(:,:,ii:jj))
    else
       call phonon_phonopy(kspace(ii:jj,:),omega(:,ii:jj),&
            velocity(:,:,ii:jj),eigenvect(:,:,ii:jj))
    end if
    call MPI_ALLREDUCE(MPI_IN_PLACE,omega,nptk*nbands,MPI_DOUBLE_PRECISION,&
         MPI_SUM,MPI_COMM_WORLD,kk)
    call MPI_ALLREDUCE(MPI_IN_PLACE,velocity,nptk*nbands*3,&
         MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,kk)
    call MPI_ALLREDUCE(MPI_IN_PLACE,eigenvect,nptk*nbands*nbands,&
         MPI_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_WORLD,kk)
    ! Make sure that group velocities have the right symmetry at each q point.
    ! This solves the problem of undefined components for degenerate modes.
    do ii=1,nptk
       velocity(:,:,ii)=matmul(symmetrizers(:,:,ii),velocity(:,:,ii))
    end do
    ! Make sure that acoustic frequencies and group velocities at Gamma
    ! are exactly zero.
    if(myid.eq.0) then
       write(*,*) "Info: about to set the acoustic frequencies at Gamma to zero"
       write(*,*) "Info: original values:"
       do ii=1,3
          write(aux,"(I1)") ii
          write(*,*) "Info: omega(1,"//aux//") =",omega(ii,1),"rad/ps"
       end do
    end if
    omega(1:3,1)=0.0_dp
    velocity(:,1:3,1)=0.0_dp
  end subroutine eigenDM

  ! Compute phonon dispersions, Phonopy style.
  subroutine phonon_phonopy(kpoints,omegas,velocities,eigenvect)
    implicit none

    real(kind=dp),intent(in) :: kpoints(:,:)
    real(kind=dp),intent(out) :: omegas(:,:),velocities(:,:,:)
    complex(kind=dp),intent(out),optional :: eigenvect(:,:,:)

    real(kind=dp),parameter :: prefactor=1745.91429109_dp ! THz^2 * amu * nm^3

    integer :: nk
    real(kind=dp),allocatable :: mm(:,:)
    complex(kind=dp),allocatable :: dyn_total(:,:),dyn_nac(:,:)
    complex(kind=dp),allocatable :: ddyn_total(:,:,:),ddyn_nac(:,:,:)
    real(kind=dp),allocatable :: fc_short(:,:,:,:,:,:,:)
    real(kind=dp),allocatable :: fc_diel(:,:,:,:,:,:,:)
    real(kind=dp),allocatable :: fc_total(:,:,:,:,:,:,:)

    integer :: i,j,ip,ik,neq
    integer :: ix1,iy1,iz1,iatom1,ix2,iy2,iz2,iatom2
    real(kind=dp) :: tmp1,tmp2,tmp3,dmin,Rnorm
    real(kind=dp) :: rcell(3),r(3),rl(3),rr(3,27),qr(27)
    complex(kind=dp) :: ztmp,star

    real(kind=dp), allocatable :: shortest(:,:)
    real(kind=dp), allocatable :: omega2(:),rwork(:)
    complex(kind=dp), allocatable :: work(:)
    integer :: nwork=1

    real(kind=dp) :: dnrm2

    nk=size(kpoints,1)

    allocate(mm(natoms,natoms))
    allocate(omega2(nbands))
    allocate(rwork(max(1,9*natoms-2)))

    allocate(fc_diel(natoms,3,scell(1),scell(2),scell(3),natoms,3))
    allocate(fc_total(natoms,3,scell(1),scell(2),scell(3),natoms,3))

    do i=1,natoms
       mm(i,i)=masses(types(i))
       do j=i+1,natoms
          mm(i,j)=sqrt(masses(types(i))*masses(types(j)))
          mm(j,i)=mm(i,j)
       end do
    end do

    ! Read FORCE_CONSTANTS_2ND and reduce the constants using mm.
    call read2fc(fc_short)

    allocate(dyn_total(nbands,nbands))
    allocate(dyn_nac(nbands,nbands))
    allocate(ddyn_total(3,nbands,nbands))
    allocate(ddyn_nac(3,nbands,nbands))
    allocate(work(nwork))
    allocate(shortest(3,nk))

    ! Use the 1st BZ image of each q point to improve the behavior of
    ! the non-analytic correction.
    do ik=1,nk
       shortest(:,ik)=kpoints(ik,:)
       tmp1=dnrm2(3,shortest(:,ik),1)
       do ix1=-2,2
          do iy1=-2,2
             do iz1=-2,2
                r=kpoints(ik,:)+ix1*rlattvec(:,1)+iy1*rlattvec(:,2)+&
                     iz1*rlattvec(:,3)
                tmp2=dnrm2(3,r,1)
                if(tmp2.lt.tmp1) then
                   tmp1=tmp2
                   shortest(:,ik)=r
                end if
             end do
          end do
       end do
    end do

    do ik=1,nk
       dyn_total=0.0_dp
       dyn_nac=0.0_dp
       ddyn_total=0.0_dp
       ddyn_nac=0.0_dp
       fc_diel=0.0_dp
       ! If the nonanalytic flag is set to TRUE, add the electrostatic
       ! correction. No correction is applied exactly at \Gamma in
       ! order not to rely on guesses about directions.
       !$OMP parallel default(none) shared(dyn_total,dyn_nac,ddyn_total,ddyn_nac,fc_diel) & 
       !$OMP & shared(nonanalytic,shortest,ik,nk,epsilon,natoms,born,mm,scell) &
       !$OMP & shared(lattvec,cartesian,fc_total,fc_short,V,kpoints) &
       !$OMP & private(tmp1,tmp2,tmp3,iatom1,iatom2,i,j,ip) &
       !$OMP & private(ix1,iy1,iz1,ix2,iy2,iz2,rcell,r,dmin,rl,Rnorm,neq,rr,qr,star,ztmp)

       if(nonanalytic.and..not.all(shortest(:,ik).eq.0.0_dp)) then
          tmp3=dot_product(shortest(:,ik),matmul(epsilon,shortest(:,ik)))
          !$OMP do collapse(4) schedule(static)
          do iatom2=1,natoms
             do j=1,3
                do iatom1=1,natoms
                   do i=1,3
                      tmp1=dot_product(shortest(:,ik),born(:,i,iatom1))
                      tmp2=dot_product(shortest(:,ik),born(:,j,iatom2))
                      dyn_nac(3*(iatom1-1)+i,3*(iatom2-1)+j)=tmp1*tmp2/&
                           mm(iatom1,iatom2)
                      ! The derivatives of the nonanalytic correction
                      ! will be needed later to make group velocities
                      ! and frequencies completely consistent.
                      do ip=1,3
                         ddyn_nac(ip,3*(iatom1-1)+i,3*(iatom2-1)+j)=&
                              tmp1*born(ip,j,iatom2)+tmp2*born(ip,i,iatom1)-&
                              2.0_dp*tmp1*tmp2*dot_product(epsilon(ip,:),shortest(:,ik))/tmp3
                      end do
                      ddyn_nac(:,3*(iatom1-1)+i,3*(iatom2-1)+j)=&
                           ddyn_nac(:,3*(iatom1-1)+i,3*(iatom2-1)+j)/&
                           mm(iatom1,iatom2)
                   end do
                end do
             end do
          end do
          !$OMP end do

          !$OMP MASTER
          dyn_nac=prefactor*dyn_nac/tmp3/V
          ddyn_nac=prefactor*ddyn_nac/tmp3/V
          !$OMP end master

          !$OMP barrier
          ! Transform back to real space to obtain a correction to the
          ! short-range force constants.
          !$OMP do collapse(4) schedule(static)
          do iatom2=1,natoms
             do j=1,3
                do iatom1=1,natoms
                   do i=1,3
                      fc_diel(iatom1,i,:,:,:,iatom2,j)=real(dyn_nac(3*(iatom1-1)+i,&
                           3*(iatom2-1)+j))
                   end do
                end do
             end do
          end do
          !$OMP end do

          !$OMP barrier
       end if

       ! Force constants with long-range correction.
       !$OMP workshare
       fc_total=fc_short+(fc_diel/real(scell(1)*scell(2)*scell(3),kind=dp))
       !$OMP end workshare

       !$OMP barrier

       ! Build the dynamical matrix and its derivatives.
       !$OMP do collapse(2) schedule(static)
       do iatom1=1,natoms
          do iatom2=1,natoms
             do ix1=1,scell(1)
                do iy1=1,scell(2)
                   do iz1=1,scell(3)
                      rcell=matmul(lattvec,(/ix1,iy1,iz1/)-(/1,1,1/))
                      r=cartesian(:,iatom1)-cartesian(:,iatom2)+rcell
                      dmin=huge(dmin)
                      do ix2=-2,2
                         do iy2=-2,2
                            do iz2=-2,2
                               rl=ix2*scell(1)*lattvec(:,1)+iy2*scell(2)*lattvec(:,2)+&
                                    iz2*scell(3)*lattvec(:,3)
                               Rnorm=dnrm2(3,rl+r,1)
                               if(abs(Rnorm-dmin).gt.1.0e-5_dp) then
                                  if(Rnorm.lt.dmin) then
                                     neq=1
                                     dmin=Rnorm
                                     qr(neq)=dot_product(kpoints(ik,:),rl+rcell)
                                     rr(:,neq)=rl+rcell
                                  endif
                               else
                                  neq=neq+1
                                  qr(neq)=dot_product(kpoints(ik,:),rl+rcell)
                                  rr(:,neq)=rl+rcell
                               endif
                            end do
                         end do
                      end do
                      star=cmplx(0.0_dp,0.0_dp,kind=dp)
                      do ip=1,neq
                         ztmp=phexp(-qr(ip))/real(neq,kind=dp)
                         star=star+ztmp
                         do i=1,3
                            do j=1,3
                               dyn_total(3*(iatom1-1)+i,3*(iatom2-1)+j)=&
                                    dyn_total(3*(iatom1-1)+i,3*(iatom2-1)+j)+&
                                    ztmp*fc_total(iatom2,j,ix1,iy1,iz1,iatom1,i)
                               ddyn_total(:,3*(iatom1-1)+i,3*(iatom2-1)+j)=&
                                    ddyn_total(:,3*(iatom1-1)+i,3*(iatom2-1)+j)-&
                                    iunit*ztmp*rr(:,ip)*fc_total(iatom2,j,ix1,iy1,iz1,iatom1,i)
                            end do
                         end do
                      end do
                      if(nonanalytic.and..not.all(kpoints(ik,:).eq.0)) then
                         do i=1,3
                            do j=1,3
                               ddyn_total(:,3*(iatom1-1)+i,3*(iatom2-1)+j)=&
                                    ddyn_total(:,3*(iatom1-1)+i,3*(iatom2-1)+j)+&
                                    star*ddyn_nac(:,3*(iatom1-1)+i,3*(iatom2-1)+j)/&
                                    real((scell(1)*scell(2)*scell(3)),kind=dp)
                            end do
                         end do
                      end if
                   end do
                end do
             end do
          end do
       end do
       !$OMP end do
       !$OMP end parallel

       ! Frequencies squared result from a diagonalization of the
       ! dynamical matrix. The first call to zheev serves to ensure that
       ! enough space has been allocated for this.
       call zheev("V","U",nbands,dyn_total,nbands,omega2,work,-1,rwork,i)

       if(real(work(1)).gt.nwork) then
          nwork=nint(2*real(work(1)))
          deallocate(work)
          allocate(work(nwork))
       end if
       call zheev("V","U",nbands,dyn_total,nbands,omega2,work,nwork,rwork,i)

       ! Eigenvectors are also returned if required.
       if(present(eigenvect)) then
          eigenvect(:,:,ik)=dyn_total
       end if

       ! As is conventional, imaginary frequencies are returned as negative.
       omegas(:,ik)=sign(sqrt(abs(omega2)),omega2)

       ! Group velocities are obtained perturbatively. This is very
       ! advatageous with respect to finite differences.
       do i=1,nbands
          do ip=1,3
             velocities(ip,i,ik)=real(dot_product(dyn_total(:,i),&
                  matmul(ddyn_total(ip,:,:),dyn_total(:,i))))
          end do
          velocities(:,i,ik)=velocities(:,i,ik)/(2.0_dp*omegas(i,ik))
       end do
    end do
    deallocate(mm,omega2,rwork,fc_short,fc_diel,fc_total,&
         dyn_total,dyn_nac,ddyn_total,ddyn_nac,work,shortest)
  end subroutine phonon_phonopy

  ! Adapted from the code of Quantum Espresso (
  ! http://www.quantum-espresso.org/ ), licensed under the GPL.
  subroutine phonon_espresso(kpoints,omegas,velocities,eigenvect)
    use iso_fortran_env, only : int64
    implicit none

    real(kind=dp),intent(in) :: kpoints(:,:)
    real(kind=dp),intent(out) :: omegas(:,:),velocities(:,:,:)
    complex(kind=dp),optional,intent(out) :: eigenvect(:,:,:)

    ! QE's 2nd-order files are in Ryd units.
    real(kind=dp),parameter :: bohr2nm=0.052917721092_dp,toTHz=20670.687_dp,&
         massfactor=1.8218779_dp*6.022e-4_dp

    integer :: ir,nreq,ntype,nat,ibrav,qscell(3)
    integer :: i,j,ipol,jpol,iat,jat,idim,jdim,t1,t2,t3,m1,m2,m3,ik
    integer :: ndim,nk,nwork,ncell_g(3)
    integer(kind=int64),allocatable :: tipo(:)
    character(len=1) :: polar_key
    character(len=5),allocatable :: label(:)
    real(kind=dp) :: weight,total_weight,exp_g,ck
    real(kind=dp) :: celldm(6),r_ws(3),rws(124,0:3),wscell(3,0:3),at(3,3)
    real(kind=dp) :: alpha,geg,gmax,kt,gr,volume_r,dnrm2
    real(kind=dp) :: cell_r(1:3,0:3),cell_g(1:3,0:3)
    real(kind=dp) :: zig(3),zjg(3),dgeg(3),t(0:3),g(0:3),g_old(0:3)
    real(kind=dp), allocatable :: omega2(:),rwork(:)
    real(kind=dp), allocatable :: k(:,:),mass(:),r(:,:),eps(:,:),mm(:,:),rr(:,:,:)
    real(kind=dp), allocatable :: zeff(:,:,:),fc_s(:,:,:,:,:,:,:)
    complex(kind=dp) :: auxi(3)
    complex(kind=dp),allocatable :: cauxiliar(:),work(:)
    complex(kind=dp),allocatable :: dyn(:,:),dyn_s(:,:,:),dyn_g(:,:,:)
    complex(kind=dp),allocatable :: ddyn(:,:,:),ddyn_s(:,:,:,:),ddyn_g(:,:,:,:)

    ! Quantum Espresso's 2nd-order format contains information about
    ! lattice vectors, atomic positions, Born effective charges and so
    ! forth in its header. The information is read but completely
    ! ignored. It is the user's responsibility to ensure that
    ! it is consistent with the CONTROL file.
    nwork=1
    nk=size(kpoints,1)
    open(1,file="espresso.ifc2",status="old")
    read(1,*) ntype,nat,ibrav,celldm(1:6)
    if (ibrav==0) then
      read(1,*) ((at(i,j),i=1,3),j=1,3)
    end if
    ntype=nelements
    nat=natoms
    ndim=3*nat

    allocate(omega2(nbands))
    allocate(work(nwork))
    allocate(rwork(max(1,9*natoms-2)))
    allocate(k(nk,3))
    allocate(label(ntype))
    allocate(mass(ntype))
    allocate(tipo(nat))
    allocate(r(nat,3))
    allocate(eps(3,3))
    allocate(zeff(nat,3,3))
    allocate(fc_s(3,3,nat,nat,scell(1),scell(2),scell(3)))
    allocate(mm(nat,nat))
    allocate(rr(nat,nat,3))
    allocate(dyn(ndim,ndim))
    allocate(dyn_s(nk,ndim,ndim))
    allocate(dyn_g(nk,ndim,ndim))
    allocate(ddyn(ndim,ndim,3))
    allocate(ddyn_s(nk,ndim,ndim,3))
    allocate(ddyn_g(nk,ndim,ndim,3))
    allocate(cauxiliar(ndim))

    do i=1,ntype
       read(1,*) j,label(i),mass(i)
    end do
    mass=masses/massfactor
    do i=1,nat
       read(1,*) j,tipo(i),r(i,1:3)
    end do
    tipo=types
    r=transpose(matmul(lattvec,positions))/bohr2nm
    read(1,*) polar_key
    if(polar_key.eq."T") then
       do i=1,3
          read(1,*) eps(i,1:3)
       end do
       do i=1,nat
          read(1,*)
          do j=1,3
             read(1,*) zeff(i,j,1:3)
          end do
       end do
    end if
    eps=transpose(epsilon)
    do i=1,nat
       zeff(i,:,:)=transpose(born(:,:,i))
    end do
    read(1,*) qscell(1:3)
    ! Read the force constants.
    do i=1,3*3*nat*nat
       read(1,*) ipol,jpol,iat,jat
       do j=1,scell(1)*scell(2)*scell(3)
          read(1,*) t1,t2,t3,fc_s(ipol,jpol,iat,jat,t1,t2,t3)
       end do
    end do
    ! Enforce the conservation of momentum in the simplest way possible.
    ! Note that this is not necessary for the Phonopy format.
    do i=1,3
       do j=1,3
          do iat=1,nat
             fc_s(i,j,iat,iat,1,1,1)=fc_s(i,j,iat,iat,1,1,1)-&
                  sum(fc_s(i,j,iat,:,:,:,:))
          end do
       end do
    end do
    close(1)

    ! Make sure operations are performed in consistent units.
    k=kpoints*bohr2nm
    cell_r(:,1:3)=transpose(lattvec)/bohr2nm
    volume_r=V/bohr2nm**3
    do i=1,3
       cell_r(i,0)=dnrm2(3,cell_r(i,1:3),1)
    end do
    cell_g(:,1:3)=transpose(rlattvec)*bohr2nm
    do i=1,3
       cell_g(i,0)=dnrm2(3,cell_g(i,1:3),1)
    end do

    ! The dynamical matrix is built in a way similar to the previous
    ! subroutine.
    wscell(1,1:3)=cell_r(1,1:3)*scell(1)
    wscell(2,1:3)=cell_r(2,1:3)*scell(2)
    wscell(3,1:3)=cell_r(3,1:3)*scell(3)

    j=1
    do m1=-2,2
       do m2=-2,2
          do m3=-2,2
             if(all((/m1,m2,m3/).eq.0)) then
                cycle
             end if
             do i=1,3
                rws(j,i)=wscell(1,i)*m1+wscell(2,i)*m2+wscell(3,i)*m3
             end do
             rws(j,0)=0.5_dp*dot_product(rws(j,1:3),rws(j,1:3))
             j=j+1
          end do
       end do
    end do

    do i=1,nat
       mm(i,i)=mass(tipo(i))
       rr(i,i,:)=0.0_dp
       do j=i+1,nat
          mm(i,j)=sqrt(mass(tipo(i))*mass(tipo(j)))
          rr(i,j,1:3)=r(i,1:3)-r(j,1:3)
          mm(j,i)=mm(i,j)
          rr(j,i,1:3)=-rr(i,j,1:3)
       end do
    end do

    gmax=14.0_dp
    alpha=(2.0_dp*pi/celldm(1))**2
    geg=gmax*4.0_dp*alpha
    ! Estimate of nrx1,nrx2,nrx3 generating all vectors up to G^2 < geg
    ! Only for dimensions where periodicity is present, e.g. if nr1=1
    ! and nr2=1, then the G-vectors run along nr3 only.
    ! (useful if system is in vacuum, e.g. 1D or 2D)
    IF (scell(1) == 1) THEN
      ncell_g(1) = 0
    ELSE
      ncell_g(1) = int ( sqrt (geg) / cell_g(1,0) ) + 1
    ENDIF
    IF (scell(2) == 1) THEN
      ncell_g(2) = 0
    ELSE
      ncell_g(2) = int ( sqrt (geg) / cell_g(2,0) ) + 1
    ENDIF
    IF (scell(3) == 1) THEN
      ncell_g(3) = 0
    ELSE
      ncell_g(3) = int ( sqrt (geg) / cell_g(3,0) ) + 1
    ENDIF

    dyn_s=0.0_dp
    ddyn_s=0.0_dp

    do iat=1,nat
       do jat=1,nat
          total_weight=0.0_dp
          do m1=-2*scell(1),2*scell(1)
             do m2=-2*scell(2),2*scell(2)
                do m3=-2*scell(3),2*scell(3)
                   do i=1,3
                      t(i)=m1*cell_r(1,i)+m2*cell_r(2,i)+m3*cell_r(3,i)
                      r_ws(i)=t(i)+rr(iat,jat,i)
                   end do
                   weight=0.0_dp
                   nreq=1
                   j=0
                   Do ir=1,124
                      ck=dot_product(r_ws,rws(ir,1:3))-rws(ir,0)
                      if(ck .gt. 1.0e-6_dp) then
                         j=1
                         cycle
                      end if
                      if(abs(ck) .lt. 1.0e-6_dp) then
                         nreq=nreq+1
                      end if
                   end do
                   if(j .eq. 0) then
                      weight=1.0_dp/real(nreq,kind=dp)
                   end if
                   if(weight.gt.0.0_dp) then
                      t1=mod(m1+1,scell(1))
                      if(t1 .le. 0) then
                         t1=t1+scell(1)
                      end if
                      t2=mod(m2+1,scell(2))
                      if(t2 .Le. 0) then
                         t2=t2+scell(2)
                      end if
                      t3=mod(m3+1,scell(3))
                      if(t3 .le. 0) then
                         t3=t3+scell(3)
                      end if
                      do ik=1,nk
                         kt=dot_product(k(ik,1:3),t(1:3))
                         do ipol=1,3
                            idim = (iat-1)*3+ipol
                            do jpol=1,3
                               jdim = (jat-1)*3+jpol
                               dyn_s(ik,idim,jdim)=dyn_s(ik,idim,jdim)+&
                                    fc_s(ipol,jpol,iat,jat,t1,t2,t3)*&
                                    phexp(-kt)*weight
                               ddyn_s(ik,idim,jdim,1:3)=ddyn_s(ik,idim,jdim,1:3)-&
                                    iunit*t(1:3)*&
                                    fc_s(ipol,jpol,iat,jat,t1,t2,t3)*&
                                    phexp(-kt)*weight
                            end do
                         end do
                      end do
                   end if
                   total_weight=total_weight+weight
                end do
             end do
          end do
       end do
    end do
    ! The nonanalytic correction has two components in this
    ! approximation. Results may differ slightly between this method
    ! and the one implemented in the previous subroutine.
    dyn_g=0.0_dp
    ddyn_g=0.0_dp
    if(nonanalytic) then
       do m1=-ncell_g(1),ncell_g(1)
          do m2=-ncell_g(2),ncell_g(2)
             do m3=-ncell_g(3),ncell_g(3)
                g(1:3)=m1*cell_g(1,1:3)+&
                     m2*cell_g(2,1:3)+m3*cell_g(3,1:3)
                geg=dot_product(g(1:3),matmul(eps,g(1:3)))
                if(geg.gt.0.0_dp.and.geg/alpha/4.0_dp.lt.gmax) then
                   exp_g=exp(-geg/alpha/4.0_dp)/geg
                   do iat=1,nat
                      zig(1:3)=matmul(g(1:3),zeff(iat,1:3,1:3))
                      auxi(1:3)=0.0_dp
                      do jat=1,nat
                         gr=dot_product(g(1:3),rr(iat,jat,1:3))
                         zjg(1:3)=matmul(g(1:3),zeff(jat,1:3,1:3))
                         auxi(1:3)=auxi(1:3)+zjg(1:3)*phexp(gr)
                      end do
                      do ipol=1,3
                         idim=(iat-1)*3+ipol
                         do jpol=1,3
                            jdim=(iat-1)*3+jpol
                            dyn_g(1:nk,idim,jdim)=dyn_g(1:nk,idim,jdim)-&
                                 exp_g*zig(ipol)*auxi(jpol)
                         end do
                      end do
                   end do
                end if
                g_old(0:3)=g(0:3)
                do ik=1,nk
                   g(1:3)=g_old(1:3)+k(ik,1:3)
                   geg=dot_product(g(1:3),matmul(eps,g(1:3)))
                   if (geg.gt.0.0_dp.and.geg/alpha/4.0_dp.lt.gmax) then
                      exp_g=exp(-geg/alpha/4.0_dp)/geg
                      dgeg=matmul(eps+transpose(eps),g(1:3))
                      do iat=1,nat
                         zig(1:3)=matmul(g(1:3),zeff(iat,1:3,1:3))
                         do jat=1,nat
                            gr=dot_product(g(1:3),rr(iat,jat,1:3))
                            zjg(1:3)=matmul(g(1:3),zeff(jat,1:3,1:3))
                            do ipol=1,3
                               idim=(iat-1)*3+ipol
                               do jpol=1,3
                                  jdim=(jat-1)*3+jpol
                                  dyn_g(ik,idim,jdim)=dyn_g(ik,idim,jdim)+&
                                       exp_g*zig(ipol)*zjg(jpol)*phexp(gr)
                                  do i=1,3
                                     ddyn_g(ik,idim,jdim,i)=ddyn_g(ik,idim,jdim,i)+&
                                          exp_g*phexp(gr)*&
                                          (zjg(jpol)*zeff(iat,i,ipol)+zig(ipol)*zeff(jat,i,jpol)+&
                                          zig(ipol)*zjg(jpol)*iunit*rr(iat,jat,i)-&
                                          zig(ipol)*zjg(jpol)*(dgeg(i)/alpha/4.0_dp+dgeg(i)/geg))
                                  end do
                               end do
                            end do
                         end do
                      end do
                   end if
                end do
             end do
          end do
       end do
       dyn_g=dyn_g*8.0_dp*pi/volume_r
       ddyn_g=ddyn_g*8.0_dp*pi/volume_r
    end if
    ! Once the dynamical matrix has been built, the frequencies and
    ! group velocities are extracted exactly like in the previous
    ! subroutine.
    do ik=1,nk
       dyn(:,:)=dyn_s(ik,:,:)+dyn_g(ik,:,:)
       ddyn(:,:,:)=ddyn_s(ik,:,:,:)+ddyn_g(ik,:,:,:)
       do ipol=1,3
          do jpol=1,3
             do iat=1,nat
                do jat=1,nat
                   idim=(iat-1)*3+ipol
                   jdim=(jat-1)*3+jpol
                   dyn(idim,jdim)=dyn(idim,jdim)/mm(iat,jat)
                   ddyn(idim,jdim,1:3)=ddyn(idim,jdim,1:3)/mm(iat,jat)
                end do
             end do
          end do
       end do

       call zheev("V","U",nbands,dyn(:,:),nbands,omega2,work,-1,rwork,i)
       if(real(work(1)).gt.nwork) then
          nwork=nint(2.0_dp*real(work(1),kind=dp))
          deallocate(work)
          allocate(work(nwork))
       end if
       call zheev("V","U",nbands,dyn(:,:),nbands,omega2,work,nwork,rwork,i)

       if(present(eigenvect)) then
          eigenvect(:,:,ik)=dyn(:,:)
       end if

       omegas(:,ik)=sign(sqrt(abs(omega2)),omega2)

       do i=1,nbands
          do j=1,3
             velocities(j,i,ik)=real(dot_product(dyn(:,i),&
                  matmul(ddyn(:,:,j),dyn(:,i))))
          end do
          velocities(:,i,ik)=velocities(:,i,ik)/(2.0_dp*omegas(i,ik))
       end do
    end do
    ! Return the result to the units used in the rest of ShengBTE.
    omegas=omegas*toTHz
    velocities=velocities*toTHz*bohr2nm
    deallocate(k)
    deallocate(label)
    deallocate(mass)
    deallocate(tipo)
    deallocate(r)
    deallocate(eps)
    deallocate(zeff)
    deallocate(fc_s)
    deallocate(mm)
    deallocate(rr)
    deallocate(dyn)
    deallocate(dyn_s)
    deallocate(dyn_g)
    deallocate(ddyn)
    deallocate(ddyn_s)
    deallocate(ddyn_g)
    deallocate(cauxiliar)
    deallocate(work)
    deallocate(rwork)
    deallocate(omega2)
  end subroutine phonon_espresso
end module phonon_routines
