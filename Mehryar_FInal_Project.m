%%Mehryar ELmi
clear all;
close all;
SubjectNumber = 2;
instructions = sprintf('find the emotions press  \n any key to con press E to ...');
path_ = cd;
path_ = [path_ '/Final project'];
files = dir (path_);
files = files(3:end);
noiseLevels = [0 30 60 100];
TimeLevels = [50,100,150,200]; %in Milliseconds
TimeLevels = TimeLevels * 1/1000; %in Seconds
VS_Levels = 100 - noiseLevels;
LN = length(noiseLevels);
LT = length(TimeLevels);
Nset = 10;
H = KbName('H');
S = KbName('S');
E = KbName('E');
Truth = horzcat(ones (1,LN*LT) , zeros(1,(LN*LT)));
responseMtx = [];
for i = 1: size (files,1)
    images(i).name= strcat('img' ,num2str(i));
    images(i).mtx = imread (strcat ( path_ , '/' , files(i).name));
    images(i).mtx = rgb2gray (imresize (images(i).mtx , [500 500]));
    for j = 1: LN     
        for p = 1:LT
            photos(LN^2*(i-1) + LT*(j-1) + p).mtx   = imnoise (images(i).mtx ,'salt & pepper', 0.01 *noiseLevels(j));
            photos(LN^2*(i-1) + LT*(j-1) + p).Time = TimeLevels(p);
            photos(LN^2*(i-1) + LT*(j-1) + p).noise =  VS_Levels(j);
            photos(LN^2*(i-1) + LT*(j-1) + p).StimulusNumber = LN^2*(i-1) + LT*(j-1) + p;
            photos(LN^2*(i-1) + LT*(j-1) + p).Truth = Truth(LN^2*(i-1) + LT*(j-1) + p);
        end
    end
end
x = [photos.StimulusNumber ; photos.noise ; photos.Time; Truth];
xx = repmat (x , 1 , Nset);
xxx = Shuffle (xx , 1);
Screen('Preference', 'SkipSyncTests', 1);
[wPtr,rect]=Screen('Openwindow',max(Screen('Screens')),[128 128 128],[0 0 1800 1000]);
[x2,  y2] = meshgrid(-128:127,  128:-1:-127);
M = 127*(1- ((y2 == 0 & x2 > -20 & x2 < 20)|(x2 == 0 & y2 > -20 & y2 < 20 ))) + 1;
fixation = Screen('MakeTexture',wPtr,M);
Xmask = imread('X.jpeg');
Xmask = rgb2gray(imresize(Xmask , [500 500]));
XmaskT = Screen('MakeTexture',wPtr,Xmask);
Ymask = imread('Y.jpeg');
Ymask = rgb2gray(imresize(Ymask , [500 500]));
YmaskT = Screen('MakeTexture',wPtr,Ymask);


for d = 1:length(xxx)
    if d == 1
        DrawFormattedText(wPtr, instructions, 'center', 'center', 1);
        Screen('Flip', wPtr);
        KbStrokeWait;
    end
    stimulustexture = Screen('MakeTexture',wPtr,photos(xxx(1,d)).mtx);
    Screen('DrawTexture',wPtr,fixation);
    Screen ('Flip',wPtr);
    WaitSecs(0.5);
    Screen ('Flip',wPtr);
    Screen ('Drawtexture',wPtr,XmaskT);
    Screen ('Flip',wPtr);
    WaitSecs(0.2);
    Screen ('Flip',wPtr);
    Screen ('DrawTexture',wPtr,stimulustexture);
    Screen ('Flip',wPtr);
    WaitSecs(photos(xxx(1,d)).Time);
    Screen ('Flip',wPtr);
    Screen ('Drawtexture',wPtr,YmaskT);
    Screen ('Flip',wPtr);
    WaitSecs(0.2);
    Screen ('Flip',wPtr);
    T = GetSecs ;
    respToBeMade = true;
    while respToBeMade &&  GetSecs < T + 1
        [keyIsDown,secs, keyCode] = KbCheck;
        if keyCode(E)
            ShowCursor;
            sca;
            return
        elseif keyCode(H)
            response = 1;
            respToBeMade = false;
            RT = secs - T; 
        elseif keyCode(S)
            response = 0;
            respToBeMade = false;
            RT = secs - T; 
            f = T + GetSecs;   
        end
    end 
    if ~keyCode
        responseMtx(d,1) = NaN ;
        responseMtx(d,5) = NaN ;
        responseMtx(d,2) = xxx(4,d); %TRUTH 
        responseMtx(d,3) = xxx(2,d); %Visual Signal
        responseMtx(d,4) = xxx(3,d); %STIMULUS TIME
    else 
        responseMtx(d,1) = response; %RESPONSE
        responseMtx(d,2) = xxx(4,d); %TRUTH 
        responseMtx(d,3) = xxx(2,d); %Visual Signal
        responseMtx(d,4) = xxx(3,d); %STIMULUS TIME
        responseMtx(d,5) = RT; %REACTION TIME
    end
end
responseMtx(any(isnan(responseMtx),2),:) = []; 
filename = strcat ('Mehryar_' , 'subject_' , num2str(SubjectNumber));
save (filename, 'responseMtx');
sca

%% Loading 
clear all;
close all;
N = 1; %enter the number of the subjet whose data is to be analyzed.
path_ = cd;
path_ = [path_ '/Mehryar_Elmi_Data'];
filename = strcat ('Mehryar_subject_', num2str(N),'.mat');
load (filename)

noiseLevels = [0 30 60 100];
TimeLevels = [50,100,150,200]; %in Milliseconds
TimeLevels = TimeLevels * 1/1000; %in Seconds
VS_Levels = 100 - noiseLevels;

%% Analysis

for m = 1:length(TimeLevels)
    for  n = 1: length(VS_Levels)
        HappyStats(n,1) = length(responseMtx(responseMtx(:,4)==TimeLevels(m) & responseMtx(:,3)== VS_Levels(n) & responseMtx(:,2)==1 ...
            & responseMtx(:,1)==1));
        HappyStats(n,2) = length(responseMtx(responseMtx(:,4)==TimeLevels(m) & responseMtx(:,3)== VS_Levels(n) & responseMtx(:,2)==1));
        HappyStats(n,3)= HappyStats(n,1)./HappyStats(n,2);
        HappyStats(n,4) = mean(responseMtx(responseMtx(:,4)==TimeLevels(m) & responseMtx(:,3)== VS_Levels(n) & responseMtx(:,2)==1 , 5));
    end
    logit_coefficient(m).Happy = glmfit(VS_Levels,[HappyStats(:,1) HappyStats(:,2)],'binomial','link','logit');
    logitFitting(m).Happy = glmval(logit_coefficient(m).Happy,VS_Levels,'logit');
end
for m = 1:length(TimeLevels)
    for  n = 1: length(VS_Levels)
        SadStats(n,1) = length(responseMtx(responseMtx(:,4)==TimeLevels(m) & responseMtx(:,3)== VS_Levels(n) & responseMtx(:,2)==0 ...
            & responseMtx(:,1)==0));
        SadStats(n,2) =  length(responseMtx(responseMtx(:,4)==TimeLevels(m) & responseMtx(:,3)== VS_Levels(n) & responseMtx(:,2)==0));
        SadStats(n,3)= SadStats(n,2)./SadStats(n,1);
        SadStats(n,4) = mean(responseMtx(responseMtx(:,4)==TimeLevels(m) & responseMtx(:,3)== VS_Levels(n) & responseMtx(:,2)==0 , 5));
    end
    logit_coefficient(m).Sad = glmfit(VS_Levels,[SadStats(:,1) SadStats(:,2)],'binomial','link','logit');
    logitFitting(m).Sad  = glmval(logit_coefficient(m).Sad,VS_Levels,'logit');
end

for s = 1:length(VS_Levels)
    for  q = 1: length(TimeLevels)
        HappyStats_(q,1) = length(responseMtx(responseMtx(:,4)==TimeLevels(q) & responseMtx(:,3)== VS_Levels(s) & responseMtx(:,2)==1 ...
            & responseMtx(:,1)==1));
        HappyStats_(q,2) = length(responseMtx(responseMtx(:,4)==TimeLevels(q) & responseMtx(:,3)== VS_Levels(s) & responseMtx(:,2)==1));
        HappyStats_(q,3)= HappyStats_(q,1)./HappyStats_(q,2);
        HappyStats_(q,4) = mean(responseMtx(responseMtx(:,4)==TimeLevels(q) & responseMtx(:,3)== VS_Levels(s) & responseMtx(:,2)==1 , 5));
    end
    logit_coefficient(s).Happy_ = glmfit(TimeLevels,[HappyStats_(:,1) HappyStats_(:,2)],'binomial','link','logit');
    logitFitting(s).Happy_ = glmval(logit_coefficient(s).Happy_,TimeLevels,'logit');
end

for s = 1:length(VS_Levels)
    for  q = 1: length(TimeLevels)
        SadStats_(q,1) = length(responseMtx(responseMtx(:,4)==TimeLevels(q) & responseMtx(:,3)== VS_Levels(s) & responseMtx(:,2)==0 ...
            & responseMtx(:,1)==0));
        SadStats_(q,2) = length(responseMtx(responseMtx(:,4)==TimeLevels(q) & responseMtx(:,3)== VS_Levels(s) & responseMtx(:,2)==0));
        SadStats_(q,3)= SadStats_(q,1)./SadStats_(q,2);
        SadStats_(q,4) = mean(responseMtx(responseMtx(:,4)==TimeLevels(q) & responseMtx(:,3)== VS_Levels(s) & responseMtx(:,2)==0 , 5));
    end
    logit_coefficient(s).Sad_ = glmfit(TimeLevels,[SadStats_(:,1) SadStats_(:,2)],'binomial','link','logit');
    logitFitting(s).Sad_ = glmval(logit_coefficient(s).Happy_,TimeLevels,'logit');
end


figure
subplot (1,2,1)
plot (TimeLevels,logitFitting(1).Happy_,TimeLevels,logitFitting(2).Happy_,TimeLevels,logitFitting(3).Happy_,TimeLevels,logitFitting(4).Happy_)
legend ('VS=100', 'VS=70', 'VS=40','VS=0','Location', 'Southeast');
title ('Happy Faces - All Visual Signals')
xlabel ('Stimulus Duration')
ylabel ('Performance')
subplot(1,2,2)
plot (TimeLevels,logitFitting(1).Sad_,TimeLevels,logitFitting(2).Sad_,TimeLevels,logitFitting(3).Sad_,TimeLevels,logitFitting(4).Sad_)
legend ('VS=100', 'VS=70', 'VS=40','VS=0','Location', 'Southeast');
title ('Happy Faces - All Visual Signals')
xlabel ('Stimulus Duration')
ylabel ('Performance')
Combined_RT_ = [HappyStats_(:,4),SadStats_(:,4)];


yyy = {'100%','70%','40%','0%'};
figure 
    bar (Combined_RT_);
    legend ('Happy','Sad', 'location', 'northeast')
    title ('Mean Reaction times - All Visual Signals')
    set(gca,'xticklabel',yyy)

figure
subplot (1,2,1)
plot (VS_Levels,logitFitting(1).Happy,VS_Levels,logitFitting(2).Happy,VS_Levels,logitFitting(3).Happy,VS_Levels,logitFitting(4).Happy)
legend ('T= 50ms', 'T=100ms', 'T=150ms','T=200ms', 'Location', 'Southeast');
title ('Happy Faces - all durations')
xlabel ('Visual Signal')
ylabel ('Performance')
subplot (1,2,2)
plot (VS_Levels,logitFitting(1).Sad,VS_Levels,logitFitting(2).Sad,VS_Levels,logitFitting(3).Sad,VS_Levels,logitFitting(4).Sad)
legend ('T= 50ms', 'T=100ms', 'T=150ms','T=200ms','Location', 'Southeast');
title ('Sad Faces - all durations')
xlabel ('Visual Signal')
ylabel ('Performance')
Combined_RT = [HappyStats(:,4),SadStats(:,4)];
timelabel = {'50ms','100ms','150ms','200ms'};

figure 
    bar (Combined_RT);
    legend ('Happy','Sad', 'location', 'northwest')
    title ('Mean Reaction times - All Durations')
    set(gca,'xticklabel',timelabel)
 