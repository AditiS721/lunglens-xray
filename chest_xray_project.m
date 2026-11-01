function chest_xray_project(action, imagePath, variant)
% CHEST_XRAY_PROJECT  Educational X-ray classification and explanation.
% Target: MATLAB R2026b, Image Processing Toolbox, Deep Learning Toolbox,
% and the MobileNet-v2 pretrained-model support package.
%
% Put the extracted dataset beneath data/raw next to this file.
% Run from MATLAB:
%   chest_xray_project("prepare")
%   chest_xray_project("train")       % trains both CLAHE and no-CLAHE
%   chest_xray_project("evaluate")    % final test evaluation and bias audit
%   chest_xray_project("explain")     % select an image interactively
%   chest_xray_project("explain", "C:/images/example.jpeg", "noclahe")
%
% Research prototype. Not for clinical use. Scores are not calibrated
% disease probabilities. Grad-CAM does not establish diagnostic validity.
% This uses inferred filename groups, NOT verified patient identifiers.
% All paths below are relative to THIS FILE, not MATLAB's current folder.

if nargin < 1, action = "prepare"; end
if nargin < 2, imagePath = ""; end
if nargin < 3, variant = "clahe"; end
root = fileparts(mfilename("fullpath"));
raw = fullfile(root,"data","raw");
models = fullfile(root,"models");
results = fullfile(root,"results");
if ~isfolder(models), mkdir(models); end
if ~isfolder(results), mkdir(results); end
splitPath = fullfile(models,"split.mat");

switch lower(string(action))
    case "prepare"
        if isfile(splitPath)
            S = load(splitPath,"T");
            fprintf("Using existing split. Delete models/split.mat only to restart the experiment.\n");
            printCounts(S.T);
            return
        end
        assert(isfolder(raw),"Create data/raw and extract the dataset there first.");
        ds = imageDatastore(raw,IncludeSubfolders=true,LabelSource="foldernames");
        files = string(ds.Files);
        labels = string(ds.Labels);
        keep = ismember(labels,["NORMAL","PNEUMONIA"]);
        files = files(keep); labels = labels(keep);
        assert(~isempty(files),"No images found inside NORMAL/PNEUMONIA folders.");
        % Sort before splitting so the seed has a stable file order.
        [files,order] = sort(files); labels = labels(order);
        groups = strings(size(files));
        for i = 1:numel(files)
            [~,name] = fileparts(files(i));
            tok = regexp(name,'^(person\d+)_','tokens','once');
            if isempty(tok)
                % Preserve NORMAL2 prefix: NORMAL2-IM-0010 differs from IM-0010.
                tok = regexp(name,'^((?:NORMAL2-)?IM-\d+)-','tokens','once');
            end
            assert(~isempty(tok), ...
                "Unknown filename: %s. Update the patient-group parser before splitting.",name);
            groups(i) = string(tok{1});
        end
        % Do not silently accept one inferred group with conflicting labels.
        u = unique(groups);
        for i = 1:numel(u)
            assert(numel(unique(labels(groups==u(i))))==1, ...
                "Group %s has conflicting labels; verify patient metadata.",u(i));
        end
        % Stratify by class, assigning entire inferred groups together.
        % This intentionally replaces the dataset's original train/val/test split.
        rng(42,"twister");
        split = strings(size(files));
        for c = ["NORMAL","PNEUMONIA"]
            ids = unique(groups(labels==c));
            assert(numel(ids)>=10,"Too few inferred patient groups for class %s.",c);
            ids = ids(randperm(numel(ids)));
            nTrain = floor(0.70*numel(ids));
            nVal = floor(0.15*numel(ids));
            split(ismember(groups,ids(1:nTrain))) = "train";
            split(ismember(groups,ids(nTrain+1:nTrain+nVal))) = "val";
            split(ismember(groups,ids(nTrain+nVal+1:end))) = "test";
        end
        T = table(files,labels,groups,split, ...
            VariableNames=["File","Label","Group","Split"]);
        assert(isempty(intersect(groups(split=="train"),groups(split=="test"))));
        assert(isempty(intersect(groups(split=="train"),groups(split=="val"))));
        assert(isempty(intersect(groups(split=="val"),groups(split=="test"))));
        save(splitPath,"T");
        writetable(T,fullfile(results,"split_manifest.csv"));
        example = T.File(find(T.Split=="train",1));
        plain = readXray(example,false,"none");
        enhanced = readXray(example,true,"none");
        f = figure; tiledlayout(2,2);
        nexttile; imshow(plain); title("Without CLAHE");
        nexttile; imshow(enhanced); title("With CLAHE");
        nexttile; imhist(plain(:,:,1)); title("Original histogram");
        nexttile; imhist(enhanced(:,:,1)); title("CLAHE histogram");
        exportgraphics(f,fullfile(results,"preprocessing.png")); close(f);
        printCounts(T);
        fprintf("Split saved. Filename groups are approximate; verify patient metadata.\n");

    case "train"
        assert(isfile(splitPath),"Run chest_xray_project('prepare') first.");
        S = load(splitPath,"T"); T = S.T;
        classNames = ["NORMAL";"PNEUMONIA"];
        batchSize = 16; % reduce to 8 or 4 if memory runs out
        epochs = 8;    % use 1 for a smoke test; 8 for the actual experiment
        for v = ["clahe","noclahe"]
            rng(42,"twister");
            useCLAHE = v=="clahe";
            trainDS = makeDS(T(T.Split=="train",:),useCLAHE,"none");
            valDS = makeDS(T(T.Split=="val",:),useCLAHE,"none");
            aug = imageDataAugmenter(RandRotation=[-10 10], ...
                RandXTranslation=[-10 10],RandYTranslation=[-10 10], ...
                RandXScale=[0.9 1.1],RandYScale=[0.9 1.1]);
            trainData = augmentedImageDatastore([224 224],trainDS,DataAugmentation=aug);
            valData = augmentedImageDatastore([224 224],valDS);
            net = imagePretrainedNetwork("mobilenetv2",NumClasses=2);
            counts = countcats(trainDS.Labels);
            assert(all(counts>0),"Both classes must appear in training.");
            weights = single(sum(counts)./(2*counts));
            % Row weights: U = singleton dimension, C = class dimension.
            lossFcn = @(Y,T) crossentropy(Y,T,weights',WeightsFormat="UC");
            options = trainingOptions("adam",InitialLearnRate=1e-4, ...
                MiniBatchSize=batchSize,MaxEpochs=epochs,Shuffle="every-epoch", ...
                ValidationData=valData, ...
                ValidationFrequency=max(1,floor(numel(trainDS.Files)/batchSize)), ...
                ValidationPatience=5,OutputNetwork="best-validation-loss", ...
                Metrics="accuracy",Plots="training-progress",Verbose=true, ...
                ExecutionEnvironment="auto");
            [net,info] = trainnet(trainData,net,lossFcn,options);
            % Save the exact split with each model for reproducible evaluation.
            save(fullfile(models,"mobilenet_"+v+".mat"), ...
                "net","info","classNames","useCLAHE","T","weights","options");
            fprintf("Saved model: %s\n",v);
        end

    case "evaluate"
        summary = table;
        for v = ["clahe","noclahe"]
            modelPath = fullfile(models,"mobilenet_"+v+".mat");
            assert(isfile(modelPath),"Missing model. Run the train command first.");
            M = load(modelPath); test = M.T(M.T.Split=="test",:);
            out = fullfile(results,v);
            if ~isfolder(out), mkdir(out); end
            ds = makeDS(test,M.useCLAHE,"none");
            scores = getScores(M.net,ds);
            [~,k] = max(scores,[],2);
            predicted = M.classNames(k);
            [row,C,fpr,tpr] = measure(test.Label,predicted,scores(:,2));
            row.Variant = v;
            summary = [summary;row]; %#ok<AGROW>
            predictions = test;
            predictions.Predicted = predicted;
            predictions.PneumoniaScore = scores(:,2);
            writetable(predictions,fullfile(out,"predictions.csv"));
            f = figure; confusionchart(C,cellstr(M.classNames));
            title("Test confusion matrix: "+v);
            exportgraphics(f,fullfile(out,"confusion_matrix.png")); close(f);
            f = figure; plot(fpr,tpr,LineWidth=2); hold on;
            plot([0 1],[0 1],"k--"); axis square; grid on;
            xlabel("False positive rate"); ylabel("Sensitivity");
            title(sprintf("%s: AUC %.3f",v,row.AUC));
            exportgraphics(f,fullfile(out,"roc.png")); close(f);

            % Perturb AFTER preprocessing, so CLAHE cannot redistribute the mask.
            % Approximate geometry only: these are not anatomical lung masks.
            audit = table("none",row.Accuracy,0, ...
                VariableNames=["Perturbation","Accuracy","AccuracyDrop"]);
            for mode = ["border","corners","center"]
                pds = makeDS(test,M.useCLAHE,mode);
                ps = getScores(M.net,pds); [~,pk] = max(ps,[],2);
                acc = mean(M.classNames(pk)==test.Label);
                audit = [audit;table(mode,acc,row.Accuracy-acc, ...
                    VariableNames=audit.Properties.VariableNames)]; %#ok<AGROW>
            end
            writetable(audit,fullfile(out,"perturbation_audit.csv"));

            % Up to 20 correct + 20 incorrect cases, sampled reproducibly.
            rng(42,"twister");
            correct = find(predicted==test.Label);
            wrong = find(predicted~=test.Label);
            chosen = [pick(correct,20);pick(wrong,20)];
            visual = table;
            for j = 1:numel(chosen)
                i = chosen(j);
                X = readXray(test.File(i),M.useCLAHE,"none");
                [map,overlay] = explainMap(M.net,X,k(i));
                stem = sprintf("case_%04d",i);
                imwrite(overlay,fullfile(out,stem+"_overlay.png"));
                imwrite(X,fullfile(out,stem+"_input.png"));
                energy = sum(map,"all");
                mask = frameMask(size(map,1),size(map,2));
                if energy>0, fraction = sum(map(mask))/energy; else, fraction = NaN; end
                areaFraction = mean(mask,"all");
                r = table(test.File(i),test.Label(i),predicted(i), ...
                    predicted(i)==test.Label(i),fraction,areaFraction, ...
                    fraction/areaFraction,"", ...
                    VariableNames=["File","TrueLabel","Predicted","Correct", ...
                    "BorderEnergy","BorderArea","BorderEnrichment","ManualCategory"]);
                visual = [visual;r]; %#ok<AGROW>
            end
            writetable(visual,fullfile(out,"visual_audit.csv"));
        end
        writetable(summary,fullfile(results,"metrics.csv")); disp(summary);
        fprintf("Results saved. Fill ManualCategory in each visual_audit.csv.\n");

    case "explain"
        variant = validatestring(variant,{'clahe','noclahe'});
        modelPath = fullfile(models,"mobilenet_"+string(variant)+".mat");
        assert(isfile(modelPath),"Train the selected model first.");
        if strlength(string(imagePath))==0
            [file,folder] = uigetfile({'*.jpeg;*.jpg;*.png;*.tif;*.tiff','X-ray images'});
            if isequal(file,0), return; end
            imagePath = fullfile(folder,file);
        end
        M = load(modelPath);
        X = readXray(imagePath,M.useCLAHE,"none");
        % MobileNet's final softmax already supplies normalized class scores.
        scores = minibatchpredict(M.net,single(X),MiniBatchSize=1);
        [score,k] = max(scores,[],2);
        [map,overlay] = explainMap(M.net,X,k);
        f = figure; tiledlayout(1,3,TileSpacing="compact");
        nexttile; imshow(imread(imagePath)); title("Original");
        nexttile; imshow(X); title("Model input");
        nexttile; imshow(overlay);
        title(sprintf("%s: score %.3f",M.classNames(k),score));
        sgtitle("Research prototype - not for clinical use");
        exportgraphics(f,fullfile(results,"latest_explanation.png"));
        imwrite(map,fullfile(results,"latest_heatmap.png"));
        fprintf("Predicted %s, model score %.3f (not calibrated confidence).\n", ...
            M.classNames(k),score);
    otherwise
        error("Use prepare, train, evaluate, or explain.");
end
end

function ds = makeDS(T,useCLAHE,mode)
ds = imageDatastore(cellstr(T.File));
ds.Labels = categorical(T.Label,["NORMAL","PNEUMONIA"]);
ds.ReadFcn = @(file) readXray(file,useCLAHE,mode);
end

function X = readXray(file,useCLAHE,mode)
I = imread(file);
if size(I,3)==3, I = rgb2gray(I); end
assert(ismatrix(I),"Expected a grayscale or RGB image: %s",file);
I = im2uint8(I);
if useCLAHE
    I = adapthisteq(I,ClipLimit=0.01,NumTiles=[8 8]);
end
I = imresize(I,[224 224]);
switch string(mode)
    case "border", I(frameMask(224,224)) = 0;
    case "corners", I(1:45,1:45) = 0; I(1:45,end-44:end) = 0;
    case "center", I(46:179,46:179) = 0;
end
X = repmat(I,[1 1 3]);
end

function mask = frameMask(h,w)
m = round(0.15*min(h,w));
mask = true(h,w); mask(m+1:end-m,m+1:end-m) = false;
end

function scores = getScores(net,ds)
data = augmentedImageDatastore([224 224],ds);
scores = minibatchpredict(net,data,MiniBatchSize=16);
assert(size(scores,1)==numel(ds.Files) && size(scores,2)==2, ...
    "Expected one row per image and two score columns.");
end

function [row,C,fpr,tpr] = measure(truth,predicted,positiveScore)
% Explicit class order avoids confusion from categorical/string conversions.
actual = truth=="PNEUMONIA"; pos = predicted=="PNEUMONIA";
TP = sum(actual & pos); FN = sum(actual & ~pos);
FP = sum(~actual & pos); TN = sum(~actual & ~pos);
C = [TN FP;FN TP];
Accuracy = mean(truth==predicted);
Sensitivity = ratio(TP,TP+FN); Specificity = ratio(TN,TN+FP);
Precision = ratio(TP,TP+FP); F1 = ratio(2*TP,2*TP+FP+FN);
% Empirical ROC with tied scores grouped; no Statistics Toolbox needed.
[s,order] = sort(positiveScore,"descend"); a = actual(order);
last = [find(diff(s)~=0);numel(s)];
tp = cumsum(a); fp = cumsum(~a);
tpr = [0;tp(last)/sum(actual)]; fpr = [0;fp(last)/sum(~actual)];
AUC = trapz(fpr,tpr);
row = table(Accuracy,Sensitivity,Specificity,Precision,F1,AUC);
end

function value = ratio(a,b)
if b==0, value = NaN; else, value = a/b; end
end

function indices = pick(indices,n)
indices = indices(randperm(numel(indices),min(n,numel(indices))));
indices = indices(:);
end

function [map,overlay] = explainMap(net,X,classIndex)
% A dlnetwork uses a numeric channel index for the target class.
% Choose final spatial feature layer automatically; report it for verification.
[map,featureLayer,reductionLayer] = gradCAM(net,single(X),classIndex, ...
    ExecutionEnvironment="cpu");
fprintf("Grad-CAM feature: %s; reduction: %s\n",featureLayer,reductionLayer);
map = imresize(max(single(map),0),[224 224],"bilinear");
peak = max(map,[],"all");
if peak>0, map = map/peak; else, map(:) = 0; end
heat = ind2rgb(uint8(255*map),jet(256));
% Spatial alpha leaves zero-importance regions unchanged.
alpha = 0.45*double(map);
overlay = (1-alpha).*im2double(X)+alpha.*heat;
end

function printCounts(T)
fprintf("Subset      NORMAL   PNEUMONIA\n");
for s = ["train","val","test"]
    fprintf("%-8s %8d %11d\n",s,sum(T.Split==s & T.Label=="NORMAL"), ...
        sum(T.Split==s & T.Label=="PNEUMONIA"));
end
end
