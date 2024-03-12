clear

%usa solo Nvectores.
%Parte 1: 
Npersones=33;
Nfacestrain=4;
Npersonstest=33;
Nfacestest=4;
Nvectors=Npersones*Nfacestrain;
OPTIONS.disp=0;
path='C:\Documents and Settings\ruben\Escritorio\Termico\s';
%[valor_x,valor_y]=distancia;
%files=640+valor_x;
%columnes=480+valor_y;
files=640;
columnes=480;
for t=1:20,
im=zeros(Npersones*Nfacestrain,files*columnes);
disp('Leyendo caras entrenamiento')
for persona=1:Npersones,
   path='C:\Documents and Settings\ruben\Escritorio\Termico\s';
   path=strcat(path,num2str(persona),'\');
   for cara=1:Nfacestrain
      path2=strcat(path,num2str(cara),'.bmt');
       [BW_vis, BW_th, out_vis, out_th] = extract_hands(path2);
       
      imatge_th=out_th;   %%agafa la imatge termica
       imatge_vis=out_vis;  %%agafa la imatge visible  
      %disp(['llegint persona ',num2str(persona),' cara ',num2str(cara)])
      %[imatge,map]=imread(path2);
         [x,y,z]= size (imatge_vis);

        if (z==1)

        else
           imatge_vis=rgb2gray(imatge_vis);
        end
        [x,y,z]= size (imatge_th);

        if (z==1)

        else
           imatge_th=rgb2gray(imatge_th);
        end
        [NormalizedHandAppearance,NormalizedHandShape,NormalizedPalm]=prepare_one_hand (imatge_vis,'left',0.7,imatge_th);
        imatge=NormalizedPalm;
      %%imatge=ait_centroid (imatge);  %%centra la imatge
      r=dct2(imatge);
      for i=1:t,
         im((persona-1)*Nfacestrain+cara,t*(i-1)+1:t*i)=r(i,1:t);
         
      end
      %im((persona-1)*Nfacestrain+cara,:)=premnmx(im((persona-1)*Nfacestrain+cara,:));
   end
end
%base=im(:,2:14*14); %quita la continua
base=im(:,1:t*t);
clear imatge;

%for Nvectors=200:-10:20,%num. maximo de vectores=cogerlos todos=Npersones*Nfacestrain;

disp('Leyendo caras test')

for person=1:Npersonstest,
   path='C:\Documents and Settings\ruben\Escritorio\Termico\s';
   path=strcat(path,num2str(person),'\');
   for face=1:Nfacestest
      path2=strcat(path,num2str(face+Nfacestrain),'.bmt'); 
      %path2=strcat(path,num2str(face),'.bmp');
      %disp(['leyendo PERSON S',num2str(person),' Reading FACE ',num2str(face+Nfacestrain)])
      %[imatge,map]=imread(path2);
      [BW_vis, BW_th, out_vis, out_th] = extract_hands(path2);
      
       imatge_th=out_th;   %%agafa la imatge termica
       imatge_vis=out_vis;  %%agafa la imatge visible 
      [x,y,z]= size (imatge_vis);

        if (z==1)

        else 
            imatge_vis=rgb2gray(imatge_vis);
        end
        [x,y,z]= size (imatge_th);

        if (z==1)

        else
           imatge_th=rgb2gray(imatge_th);
        end
      [NormalizedHandAppearance,NormalizedHandShape,NormalizedPalm]=prepare_one_hand (imatge_vis,'left',0.7,imatge_th);
        imatge=NormalizedPalm;
      %%imatge=ait_centroid (imatge);  %%centra la imatge
      r=dct2(imatge);
      for i=1:t,
         MatriuZtest((person-1)*Nfacestest+face,t*(i-1)+1:t*i)=r(i,1:t);
      end
      %Normalizació dels valors de la matriu matriuZ a valors [-1 1]
      %MatriuZtest((person-1)*Nfacestest+face,:)=premnmx(MatriuZtest((person-1)*Nfacestest+face,:));
   end
end
%test=MatriuZtest(:,2:14*14);
test=MatriuZtest(:,1:t*t);

clear MatriuZtest;



%Parte 5:
for i=1:Npersonstest*Nfacestest,
   for j=1:Npersones*Nfacestrain,
      distancia(j,i)= mean(abs(base(j,:)-test(i,:)));
   end
end
encert=0;errada=0;
Nfaces=Nfacestrain;
for persons=1:Npersonstest,
   for faces=1:Nfacestest,
      [valor,posicio]=min(distancia(:,(persons-1)*Nfacestest+faces));
      %posicio
   if (posicio> (persons-1)*Nfaces) &&(posicio<(persons-1)*Nfaces+Nfaces+1 ), encert=encert+1;
   else errada=errada+1;
   end
   end
end
   
   taxa(t)=encert/(encert+errada)*100;
   disp(['percentatge de reconeixement ',num2str(taxa(t))])
%   tasa(iteracion)=taxa;
 %  iteracion=iteracion+1;
   
   %end %del for Nvectors

%plot(200:-10:20,tasa)
%grid
%xlabel('Number of eigenvectors')
%ylabel('recognition rate (%)')
end