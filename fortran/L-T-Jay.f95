program converse
implicit none
real matter,w,zmin,zmax,lmin,lmax,bmin,bmax,pi,d_L,eruth,a,b,logLmodel,Ho,E_z,Tmin_1,Tmax_1,mean_err,intr_scat
real Lmodel,a1,b1,matter_samp,w_samp,Ho_samp,logL_th,Tmodel,first,T_function,a_plus,b_plus,a_minus,b_minus,sk,stat_unc
real l_pol, b_pol, theta, theta_given, costh,costh2,rad,l_pol2,b_pol2,theta2,theta1,z_corr,dL,e_dL,distmod,e_distmod
real,dimension(400):: T,nH,flux,z,glon,glat,sigma_logT,logT,Tmin,Tmax,resid_logL,ra,dec,eL,sig_eL,metal,metal_max,metal_min,nH_H1
real,dimension(400):: metal_core,bcg_offset,rass,Lum_frac,glon1,glat1,sz_offset,Y,eY,resid_logLbcg,stat_unc_bcg,L_bcg
integer i,io,u,p,j,crit,crit2,flag,flag2,metr
double precision dL_cm_mine,dL_cm_sample,minx,minx1
double precision,dimension(400):: L_z,Lum,logL_z
double precision,dimension(8001,7001):: x_tetr,diaf_x
character(len=25),dimension(400):: name,catal,telesc,name2

!Lum=L_bol from data, L_th=4πd_L^2 x given flux, Lmodel=aT^b

pi=3.1415927

matter=0.3
w=-1
Ho=70.


io=0.

 zmin=0
 zmax=2


print*, 'Give T_x range in keV'
read*, Tmin_1,Tmax_1


open(unit=1,file='master_file.txt',status='old')

read(1,*) !this is just to skip the first row with the column names



print*, "Give Glon and Glot"

read(*,*) l_pol, b_pol



l_pol=l_pol*pi/180
b_pol=b_pol*pi/180



print*, "Give radius in degrees"

read(*,*) theta_given
io=0.


do i=1,1000


 read(1,*,end=123) name(i), z(i),glon(i),glat(i),sz_offset(i),T(i),Tmax(i),Tmin(i),Lum(i),eL(i)


glon1(i)=glon(i)*pi/180 
glat1(i)=glat(i)*pi/180 


costh=cos(pi/2-b_pol)*cos(pi/2-glat1(i))+sin(pi/2-b_pol)*sin(pi/2-glat1(i))*cos(glon1(i)-l_pol)     ! cosθ=cosa*cosb+sina*sinb*cosA
theta=acos(costh)
theta1=theta*180/pi




if ((z(i)>zmin .and. z(i)<=zmax) &
.and. (T(i)<=Tmax_1 .and. T(i)>Tmin_1).and. theta1<=theta_given) then
	 io=io+1
	 name(io)=name(i)

z(io)=z(i)
T(io)=(T(i)/4.)
Tmin(io)=(Tmin(i)/4.)
Tmax(io)=(Tmax(i)/4.)

sigma_logT(io)=0.4343*(Tmax(io)-Tmin(io))/(2*T(io))!/costh , the costh is for when you do the anisotropy analysis, ignore it for now
glon(io)=glon(i)
glat(io)=glat(i)
Lum(io)=Lum(i)

eL(io)=eL(i)
sig_eL(io)=(0.4343*eL(i)/100)!/costh 


endif
end do


123 continue

print*, io

do i=1,io

							 
!Correcting L for cosmology of sample and redshift evolution Lbol/E(z) and dividing with  10^44                                                     
E_z=sqrt(matter*(1+z(i))**3+(1-matter))
L_z(i)=Lum(i)/E_z

logL_z(i)=log10(L_z(i))
logT(i)=log10(T(i))
end do


minx1=5 !arbtirary large number for reduced chi square which should be ~1

intr_scat=0.15  !starting minimum value of intrinsic scatter term, which is simply extra "intrinsic" uncertainty of the scaling relation due to dispersion

do while (minx1>1.04)  !run the fits until you reach the minimum intr_scat that gives you a reduced chi square ~1, as it should be for a good fit
	intr_scat=intr_scat+0.007

	print*, intr_scat


	a=-0.1  !initial grid value of normalization where logL=a+b*logT
	b=1.5  !same for slope
	minx=10000000.  !some large number for comparison later
	do u=1,1000   !how many steps of slope values I want

		do p=1,500   !same for normalization
			x_tetr(u,p)=0   !I have a table here for reasons irrelevant for you (since I removed this part from the code I'm sending you, so you just need a float number
			
			do j=1,io   !runs for each of the io clusters

				x_tetr(u,p)=x_tetr(u,p)+((logL_z(j)-a-b*logT(j))**2)/((sigma_logT(j)*b)**2+sig_eL(j)**2+intr_scat**2)  !add the chi-square that corresponds to each cluster to the existing one, until you sum for all clusters
			
			end do

			if(x_tetr(u,p)<=minx) then  !if the final chi-square (x_tert) is smaller than the minx (initial large number), save the a and b you used, and replace minx with the current chi-square to find better models in the following iterations
			minx=x_tetr(u,p)
			a1=a
			b1=b
			end if
				
			a=a+0.001  !increase step of normalization
		end do  !finish all normalization steps for the first slope value
		a=-0.1
		b=b+0.002  !increase step of slope
	end do
	minx1=minx/(io-3.)  !reduced chi square is total chi square divided by no. of clusters minus free parameters

	print*, minx1, 10**(a1), b1  !just print best-fit results for each intrinsic scatter term. This is just to keep track of the code, not needed

end do

write(*,*) a1,b1,minx,minx1,'            ', '10^a1=factor/10^44=', 10**a1, 'intr_scat=', intr_scat  !print final best-fit results

close(1)


end program converse




!The following subroutine is to change redshifts and cluster properties for a given peculiar velocity (amplitude and direction). This is irrelevant for now and you can ignore it. Not needed for the default fit. After a few months we can also use this so I just leave it here for now      

subroutine lum_dist(z,l,b,matter,w,Ho,d_L,z_corr)
real f,z,matter,w,d_L,dist_modu,h,timh,o,eryth,Ho,lx,bx,l,b,c,u,pi,costh,z_corr,u_c_correct
integer p

pi=3.14159
lx=264.
c=299792.453
bx=48.
o=0.
eryth=0.
bx=bx*pi/180
b=b*pi/180 
lx=lx*pi/180
l=l*pi/180 
costh=cos(pi/2-bx)*cos(pi/2-b)+sin(pi/2-bx)*sin(pi/2-b)*cos(l-lx)    ! cosθ=cosa*cosb+sina*sinb*cosA
u=369*costh     
u_c_correct=((1+z)**2-1)/((1+z)**2+1)+u/c
z_corr=sqrt((1+u_c_correct)/(1-u_c_correct))-1
!z_corr=z+u/c
!print*, costh, u_c_correct,z_corr,z
h=abs((z_corr)/100.)

!if (b .ne. 0) then
!write(47,*) l*180/pi,b*180/pi,u,costh, acos(costh)*180/pi,z,z_corr
!end if

timh=(f(o,matter,w)+f((z_corr),matter,w))*(h/3.)
do p=1,99
eryth=eryth+h
if (mod(p,2)==1) then
timh=timh+4.*f(eryth,matter,w)*(h/3.)
else
timh=timh+2.*f(eryth,matter,w)*(h/3.)
end if
end do
d_L=timh*(299792.453*(1+z_corr)/Ho)                             !luminosity distance=c(1+z)/Ho * oloklirwma
dist_modu=5*log10(apostasi)+25                                !μ=5log(apostasi)+25
end subroutine lum_dist


subroutine lum_dist_samp(z,matter,w,Ho,d_L)
real f,z,matter,w,d_L,dist_modu,h,timh,o,eryth,Ho,lx,bx,l,b,c,u,pi
integer p


o=0.
eryth=0.
h=abs((z)/100.)

timh=(f(o,matter,w)+f((z),matter,w))*(h/3.)
do p=1,99
eryth=eryth+h
if (mod(p,2)==1) then
timh=timh+4.*f(eryth,matter,w)*(h/3.)
else
timh=timh+2.*f(eryth,matter,w)*(h/3.)
end if
end do
d_L=timh*(299792.453*(1+z)/Ho)                             !luminosity distance=c(1+z)/Ho * oloklirwma
dist_modu=5*log10(apostasi)+25                                !μ=5log(apostasi)+25
end subroutine lum_dist_samp




real function rad(x)
				real x
				rad=x*3.14159268/180

end function

