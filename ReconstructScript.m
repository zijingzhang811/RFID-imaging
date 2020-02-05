close all
clear
clc

% Parameters.
[tagPosition, rxPosition, freq] = tag_antenna_positions3D_func();
TagNum = size(tagPosition, 1);
RecvNum = size(rxPosition, 1);
RecvNum0=4;
FreqNum = length(freq);
c = physconst('LightSpeed'); 

%% Data Collection.

% Threshold for standard deviation of phase.
phStdTh = 15; % In degrees. A very high value means no rejection.
% fprintf('Using phase standard deviation threshold %d\n', phStdTh);
rssiDiffTh = .5;    % Set the threshold of RSSI. This variable is useless is you set "RJT" equal to 0.
RJT = 1;    % 1 means you threshold the RSSI, and other values means you don't threshold the RSSI.

% Specify data directory and name.
dirName = 'F:\RFID imaging\data 0112720\';
% fileName = 'wo_1224';
opts.calibType = 1;

% Load the raw data. wo 1-1
load([dirName,  'data_wo_0127','.mat']); % Looking at 2 minute cases  wo_1224
% data_wo = [chindexlist, tagindexlist, antennalist, rssiimpinjlist, rssiimpinjlist_d, phasedeglist];

% Generate the data matrices.
PhaseDataCollect11 = cell(TagNum, RecvNum0, FreqNum);
PhaseCollectStd11 = zeros(TagNum, RecvNum0, FreqNum);
PhaseCollectMean11 = zeros(TagNum, RecvNum0, FreqNum);
RSSIDataCollect11 = cell(TagNum, RecvNum0, FreqNum);
RSSICollectStd111 = zeros(TagNum, RecvNum0, FreqNum);
RSSICollectMean11 = zeros(TagNum, RecvNum0, FreqNum);
CompNumDataCollect11 = cell(TagNum, RecvNum0, FreqNum);
CompNumCollectMean11 = zeros(TagNum, RecvNum0, FreqNum);
for j = 1:RecvNum0
    phase_t = phasedeglist(antennalist == j);
    rssi_t = rssiimpinjlist_d(antennalist == j);
    cha_t = chindexlist(antennalist == j);
    tag_t = tagindexlist(antennalist == j);
    for k = 1:FreqNum
        phase_tt = phase_t(cha_t == k);
        rssi_tt = rssi_t(cha_t == k);
        tag_tt = tag_t(cha_t == k);
        for i = 1:TagNum
            phase_ttt = phase_tt(tag_tt == i);
            phase_ttt = rad2deg(unwrap(deg2rad(phase_ttt)));
            rssi_ttt = rssi_tt(tag_tt == i);
            if ~isempty(phase_ttt)
                PhaseDataCollect11{i, j, k} = phase_ttt;
                PhaseCollectStd11(i, j, k) = std(phase_ttt);
                RSSIDataCollect11{i, j, k} = rssi_ttt;
                RSSICollectStd111(i, j, k) = std(rssi_ttt);
                CompNumDataCollect11{i, j, k} = rssi_ttt.*cos(deg2rad(phase_ttt)) + sqrt(-1).*rssi_ttt.*sin(deg2rad(phase_ttt));
                if PhaseCollectStd11(i, j, k) > phStdTh
                    % First completely reject the channel using phase
                    % standard deviation threshold.
                    PhaseCollectMean11(i, j, k) = NaN;
                    RSSICollectMean11(i, j, k) = NaN;
                    CompNumCollectMean11(i, j, k) = NaN;
                else
                    PhaseCollectMean11(i, j, k) = mean(phase_ttt);
                    RSSICollectMean11(i, j, k) = mean(rssi_ttt);
                    CompNumCollectMean11(i, j, k) = mean(CompNumDataCollect11{i, j, k});
                end
            else
                % Simply there's no data received for this channel.
                PhaseDataCollect11{i, j, k} = NaN;
                PhaseCollectStd11(i, j, k) = NaN;
                PhaseCollectMean11(i, j, k) = NaN;
                RSSIDataCollect11{i, j, k} = NaN;
                RSSICollectStd111(i, j, k) = NaN;
                RSSICollectMean11(i, j, k) = NaN;
                CompNumDataCollect11{i, j, k} = NaN;
                CompNumCollectMean11(i, j, k) = NaN;
            end                
        end
    end
end
clear chindexlist tagindexlist antennalist rssiimpinjlist rssiimpinjlist_d phasedeglist
% Load the raw data. wo 1-2
load([dirName, 'data_wo02_0127','.mat']); % Looking at 2 minute cases
% data_wo = [chindexlist, tagindexlist, antennalist, rssiimpinjlist, rssiimpinjlist_d, phasedeglist];

% Generate the data matrices.
PhaseDataCollect12 = cell(TagNum, RecvNum0, FreqNum);
PhaseCollectStd12 = zeros(TagNum, RecvNum0, FreqNum);
PhaseCollectMean12 = zeros(TagNum, RecvNum0, FreqNum);
RSSIDataCollect12 = cell(TagNum, RecvNum0, FreqNum);
RSSICollectStd112 = zeros(TagNum, RecvNum0, FreqNum);
RSSICollectMean12 = zeros(TagNum, RecvNum0, FreqNum);
CompNumDataCollect12 = cell(TagNum, RecvNum0, FreqNum);
CompNumCollectMean12 = zeros(TagNum, RecvNum0, FreqNum);
for j = 1:RecvNum0
    phase_t = phasedeglist(antennalist == j);
    rssi_t = rssiimpinjlist_d(antennalist == j);
    cha_t = chindexlist(antennalist == j);
    tag_t = tagindexlist(antennalist == j);
    for k = 1:FreqNum
        phase_tt = phase_t(cha_t == k);
        rssi_tt = rssi_t(cha_t == k);
        tag_tt = tag_t(cha_t == k);
        for i = 1:TagNum
            phase_ttt = phase_tt(tag_tt == i);
            phase_ttt = rad2deg(unwrap(deg2rad(phase_ttt)));
            rssi_ttt = rssi_tt(tag_tt == i);
            if ~isempty(phase_ttt)
                PhaseDataCollect12{i, j, k} = phase_ttt;
                PhaseCollectStd12(i, j, k) = std(phase_ttt);
                RSSIDataCollect12{i, j, k} = rssi_ttt;
                RSSICollectStd112(i, j, k) = std(rssi_ttt);
                CompNumDataCollect12{i, j, k} = rssi_ttt.*cos(deg2rad(phase_ttt)) + sqrt(-1).*rssi_ttt.*sin(deg2rad(phase_ttt));
                if PhaseCollectStd12(i, j, k) > phStdTh
                    % First completely reject the channel using phase
                    % standard deviation threshold.
                    PhaseCollectMean12(i, j, k) = NaN;
                    RSSICollectMean12(i, j, k) = NaN;
                    CompNumCollectMean12(i, j, k) = NaN;
                else
                    PhaseCollectMean12(i, j, k) = mean(phase_ttt);
                    RSSICollectMean12(i, j, k) = mean(rssi_ttt);
                    CompNumCollectMean12(i, j, k) = mean(CompNumDataCollect12{i, j, k});
                end
            else
                % Simply there's no data received for this channel.
                PhaseDataCollect12{i, j, k} = NaN;
                PhaseCollectStd12(i, j, k) = NaN;
                PhaseCollectMean12(i, j, k) = NaN;
                RSSIDataCollect12{i, j, k} = NaN;
                RSSICollectStd112(i, j, k) = NaN;
                RSSICollectMean12(i, j, k) = NaN;
                CompNumDataCollect12{i, j, k} = NaN;
                CompNumCollectMean12(i, j, k) = NaN;
            end                
        end
    end
end
clear chindexlist tagindexlist antennalist rssiimpinjlist rssiimpinjlist_d phasedeglist


% Load the raw data.  w 2-1 smallball02_1224
load([dirName,'data_zijing_0127','.mat']); 
% data_w = [chindexlist, tagindexlist, antennalist, rssiimpinjlist, rssiimpinjlist_d, phasedeglist];

% Generate the data matrices.
PhaseDataCollect21 = cell(TagNum, RecvNum0, FreqNum);
PhaseCollectStd21 = zeros(TagNum, RecvNum0, FreqNum);
PhaseCollectMean21 = zeros(TagNum, RecvNum0, FreqNum);
RSSIDataCollect21 = cell(TagNum, RecvNum0, FreqNum);
RSSICollectStd21 = zeros(TagNum, RecvNum0, FreqNum);
RSSICollectMean21 = zeros(TagNum, RecvNum0, FreqNum);
CompNumDataCollect21 = cell(TagNum, RecvNum0, FreqNum);
CompNumCollectMean21 = zeros(TagNum, RecvNum0, FreqNum);
for j = 1:RecvNum0
    phase_t = phasedeglist(antennalist == j);
    rssi_t = rssiimpinjlist_d(antennalist == j);
    cha_t = chindexlist(antennalist == j);
    tag_t = tagindexlist(antennalist == j);
    for k = 1:FreqNum
        phase_tt = phase_t(cha_t == k);
        rssi_tt = rssi_t(cha_t == k);
        tag_tt = tag_t(cha_t == k);
        for i = 1:TagNum
            phase_ttt = phase_tt(tag_tt == i);
            phase_ttt = rad2deg(unwrap(deg2rad(phase_ttt)));
            rssi_ttt = rssi_tt(tag_tt == i);
            if ~isempty(phase_ttt)
                PhaseDataCollect21{i, j, k} = phase_ttt;
                PhaseCollectStd21(i, j, k) = std(phase_ttt);
                RSSIDataCollect21{i, j, k} = rssi_ttt;
                RSSICollectStd21(i, j, k) = std(rssi_ttt);
                CompNumDataCollect21{i, j, k} = rssi_ttt.*cos(deg2rad(phase_ttt)) + sqrt(-1).*rssi_ttt.*sin(deg2rad(phase_ttt));
                if PhaseCollectStd21(i, j, k) > phStdTh
                    % First completely reject the channel using phase
                    % standard deviation threshold.
                    PhaseCollectMean21(i, j, k) = NaN;
                    RSSICollectMean21(i, j, k) = NaN;
                    CompNumCollectMean21(i, j, k) = NaN;
                else
                    PhaseCollectMean21(i, j, k) = mean(phase_ttt);
                    RSSICollectMean21(i, j, k) = mean(rssi_ttt);
                    CompNumCollectMean21(i, j, k) = mean(CompNumDataCollect21{i, j, k});
                end
            else
                % Simply there's no data received for this channel.
                PhaseDataCollect21{i, j, k} = NaN;
                PhaseCollectStd21(i, j, k) = NaN;
                PhaseCollectMean21(i, j, k) = NaN;
                RSSIDataCollect21{i, j, k} = NaN;
                RSSICollectStd21(i, j, k) = NaN;
                RSSICollectMean21(i, j, k) = NaN;
                CompNumDataCollect21{i, j, k} = NaN;
                CompNumCollectMean21(i, j, k) = NaN;
            end                
        end
    end
end
clear chindexlist tagindexlist antennalist rssiimpinjlist rssiimpinjlist_d phasedeglist

% Load the raw data.  w 2-2
load([dirName, 'data_zijing02_0127','.mat']); 
% data_w = [chindexlist, tagindexlist, antennalist, rssiimpinjlist, rssiimpinjlist_d, phasedeglist];

% Generate the data matrices.
PhaseDataCollect22 = cell(TagNum, RecvNum0, FreqNum);
PhaseCollectStd22 = zeros(TagNum, RecvNum0, FreqNum);
PhaseCollectMean22 = zeros(TagNum, RecvNum0, FreqNum);
RSSIDataCollect22 = cell(TagNum, RecvNum0, FreqNum);
RSSICollectStd22 = zeros(TagNum, RecvNum0, FreqNum);
RSSICollectMean22 = zeros(TagNum, RecvNum0, FreqNum);
CompNumDataCollect22 = cell(TagNum, RecvNum0, FreqNum);
CompNumCollectMean22 = zeros(TagNum, RecvNum0, FreqNum);
for j = 1:RecvNum0
    phase_t = phasedeglist(antennalist == j);
    rssi_t = rssiimpinjlist_d(antennalist == j);
    cha_t = chindexlist(antennalist == j);
    tag_t = tagindexlist(antennalist == j);
    for k = 1:FreqNum
        phase_tt = phase_t(cha_t == k);
        rssi_tt = rssi_t(cha_t == k);
        tag_tt = tag_t(cha_t == k);
        for i = 1:TagNum
            phase_ttt = phase_tt(tag_tt == i);
            phase_ttt = rad2deg(unwrap(deg2rad(phase_ttt)));
            rssi_ttt = rssi_tt(tag_tt == i);
            if ~isempty(phase_ttt)
                PhaseDataCollect22{i, j, k} = phase_ttt;
                PhaseCollectStd22(i, j, k) = std(phase_ttt);
                RSSIDataCollect22{i, j, k} = rssi_ttt;
                RSSICollectStd22(i, j, k) = std(rssi_ttt);
                CompNumDataCollect22{i, j, k} = rssi_ttt.*cos(deg2rad(phase_ttt)) + sqrt(-1).*rssi_ttt.*sin(deg2rad(phase_ttt));
                if PhaseCollectStd22(i, j, k) > phStdTh
                    % First completely reject the channel using phase
                    % standard deviation threshold.
                    PhaseCollectMean22(i, j, k) = NaN;
                    RSSICollectMean22(i, j, k) = NaN;
                    CompNumCollectMean22(i, j, k) = NaN;
                else
                    PhaseCollectMean22(i, j, k) = mean(phase_ttt);
                    RSSICollectMean22(i, j, k) = mean(rssi_ttt);
                    CompNumCollectMean22(i, j, k) = mean(CompNumDataCollect22{i, j, k});
                end
            else
                % Simply there's no data received for this channel.
                PhaseDataCollect22{i, j, k} = NaN;
                PhaseCollectStd22(i, j, k) = NaN;
                PhaseCollectMean22(i, j, k) = NaN;
                RSSIDataCollect22{i, j, k} = NaN;
                RSSICollectStd22(i, j, k) = NaN;
                RSSICollectMean22(i, j, k) = NaN;
                CompNumDataCollect22{i, j, k} = NaN;
                CompNumCollectMean22(i, j, k) = NaN;
            end                
        end
    end
end
clear chindexlist tagindexlist antennalist rssiimpinjlist rssiimpinjlist_d phasedeglist
RSSICollectMean1=cat(2,RSSICollectMean11,RSSICollectMean12);
RSSICollectMean2=cat(2,RSSICollectMean21,RSSICollectMean22);
CompNumCollectMean1=cat(2,CompNumCollectMean11,CompNumCollectMean12);
CompNumCollectMean2=cat(2,CompNumCollectMean21,CompNumCollectMean22);
PhaseCollectMean1=cat(2,PhaseCollectMean11,PhaseCollectMean12);
PhaseCollectMean2=cat(2,PhaseCollectMean21,PhaseCollectMean22);
% Reject the channels with blocked RSSI's.
if RJT == 1
    for n = 1:TagNum
        for k = 1:RecvNum
            for m = 1:FreqNum
                if RSSICollectMean2(n, k, m)/RSSICollectMean1(n, k, m) < rssiDiffTh
                    RSSICollectMean1(n, k, m) = NaN;
                    PhaseCollectMean1(n, k, m) = NaN;
                    CompNumCollectMean1(n, k, m) = NaN;
                    RSSICollectMean2(n, k, m) = NaN;
                    PhaseCollectMean2(n, k, m) = NaN;
                    CompNumCollectMean2(n, k, m) = NaN;
                end
            end
        end
    end
end


G_wo_s = CompNumCollectMean1;
G_w_s = CompNumCollectMean2;

% Find the zero values in the calibration matrix, and replace them with the
% maximum value in the matrix. (We can replace those zeros with any number
% we like, actually.)
idx_wo = find(RSSICollectMean1 ~= RSSICollectMean1);
idx_w = find(RSSICollectMean2 ~= RSSICollectMean2);
idx = [idx_wo; idx_w];
idx = unique(idx);
% ConstParam = max(max(max(abs(G_wo_s))));
ConstParam = 0;
% G_wo_s(idx_wo) = ConstParam;
% G_w_s(idx_wo) = ConstParam;
% G_w_s(idx_w) = G_wo_s(idx_w);
G_wo_s(idx) = ConstParam;
G_w_s(idx) = ConstParam;

% How many channels are lost.
NumLost = length(idx);
PercLost = NumLost/length(G_wo_s(:));
fprintf('Percentage Tx-Rx-Freq combination lost: %3.2f%% \n', 100*PercLost);

%% Calibration.
% Calculate the distance from tags to receivers.
r_x = bsxfun(@minus, tagPosition(:, 1), rxPosition(:, 1)');
r_y = bsxfun(@minus, tagPosition(:, 2), rxPosition(:, 2)');
r_z = bsxfun(@minus, tagPosition(:, 3), rxPosition(:, 3)');
r = sqrt(r_x.^2 + r_y.^2 + r_z.^2);

% Perform the calibration.
G_calib = zeros(TagNum, RecvNum, FreqNum);
switch(opts.calibType)
    case 1
        % Differential receiving: First calculate the ratio of the data
        % received by one receiver antenna, to the data received by all the
        % other receiver antennas. Perform this for the data from both the
        % case when no object is present, and the case when object is
        % present. Then calculate the ratio of the above-calculated ratios
        % of the case when object is present to the case when no object is
        % present. The results are stored in the matrix named G_calib.
        for m = 1:FreqNum
            for n = 1:TagNum
                for k = 1:RecvNum
                    for k_pair = 1:RecvNum
                        if k_pair ~= k
                            if G_w_s(n, k_pair, m) ~= 0 && G_wo_s(n, k_pair, m) ~= 0 ...
                                    && G_w_s(n, k, m) ~= 0 && G_wo_s(n, k, m) ~= 0
                                % Both denominators and both numerators
                                % should not be zero.
                                G_wo_s_ratio = G_wo_s(n, k, m)/G_wo_s(n, k_pair, m);
                                G_w_s_ratio = G_w_s(n, k, m)/G_w_s(n, k_pair, m);
                                G_calib(n, k, m) = G_calib(n, k, m) + (G_w_s_ratio/G_wo_s_ratio - 1) ...
                                    *exp(-1j*(2*pi*freq(m)/c)*r(n, k));
                            else
                                G_calib(n, k, m) = G_calib(n, k, m);
                            end
                        end
                    end
                end
            end
        end
        
        % Eliminate the weighting part in Tag phase. (calibration 3)
        % For each receiver, how the tag phase is changing. (calibration 4)
        % consider RF cable and clutter as a whole. (Guoyi) (calibration 5)
        % Instead of using receiver 1 as reference, what if we use another
        % receiver as reference?
    case 2
        % Another calibration method: First calculate the RF cable and
        % clutter loss (phase offset). Then calculate the loss introduced
        % by each tag. Then we subtract these two parts from the raw data
        % we collected.
        
        % Calculate the RF cable and clutter loss (phase offset). We need
        % to use the data from the case when no object is present, so that
        % after subtracting the LoS propagation phase offset, we're left
        % with only the RF cable and clutter loss. Here we calculate the
        % differential loss with reference to the first receiving antenna.
        scale = 1e-3;
        G_w_s = scale.*G_w_s;
        G_wo_s = scale.*G_wo_s;
        CableClutter = zeros(TagNum, RecvNum, FreqNum);
        CableClutter(:, 1, :) = 1;
        for m = 1:FreqNum
            kn = 2*pi*freq(m)/c;
            for k = 1:RecvNum
                for n = 1:TagNum
                    distTagRecv1 = norm(tagPosition(n, :) - rxPosition(1, :));
                    distTagRecv = norm(tagPosition(n, :) - rxPosition(k, :));
                    RelPhs = -kn*(distTagRecv - distTagRecv1);
                    % comment: why not 2 times the distance?
                    RelCompx = cos(RelPhs) + 1j*sin(RelPhs);
%                     RelCompx = exp(1i*RelPhs);
                    RelRawData = G_wo_s(n, k, m)*G_wo_s(n, 1, m)';
                    CableClutter(n, k, m) = RelCompx*RelRawData';
                    if abs(CableClutter(n, k, m)) > 0
                        CableClutter(n, k, m) = CableClutter(n, k, m)./abs(CableClutter(n, k, m));
                    end
                end
            end
        end
        
        % Calculate the phase offset introduced by each tag at each
        % frequency. Here we're calculating the tag loss averaging on all
        % the receiving antenna.
%                 TagCal = ones(FreqNum, TagNum);
        TagCal = zeros(FreqNum, TagNum);
        for m = 1:FreqNum
            kn = 2*pi*freq(m)/c;
            for n = 1:TagNum
                sumComp = 0;
                for k = 1:RecvNum
                    distTagRecv = norm(tagPosition(n, :) - rxPosition(k, :));
                    RelPhs = -kn*distTagRecv;
                    % comment: why not 2 times the distance?
                    RelCompx = cos(RelPhs) + 1j*sin(RelPhs);
%                     RelCompx = exp(1i*RelPhs);
                    RelRawData = G_wo_s(n, k, m)*CableClutter(n, k, m);
                    sumComp = sumComp + RelCompx*RelRawData';
                end
                if abs(sumComp) > 0
                    TagCal(m, n) = sumComp/abs(sumComp);
                else
                    TagCal(m, n) = 1;
                end
            end
        end
        
        countBlocked = 0;
        for n = 1:TagNum
            for k = 1:RecvNum
                G_w_abs = abs(G_w_s(n, k, :));
                G_wo_abs = abs(G_wo_s(n, k, :));
                isLosBlocked = 1;
                for m = 1:FreqNum
                    if G_w_abs(m)*G_wo_abs(m) > 0
                        if G_w_abs(m)/G_wo_abs(m) < rssiDiffTh
                            isLosBLocked = 1*isLosBlocked;
                        else
                            isLosBlocked = 0*isLosBlocked;
                        end
                    end
                end
                
                for m = 1:FreqNum
                    if G_w_abs(m)*G_wo_abs(m) > 0
                        if isLosBlocked == 0
                            G_calib(n, k, m) = G_w_s(n, k, m) - G_wo_s(n, k, m);
                            fac = CableClutter(n, k, m)*TagCal(m, n);
                            G_calib(n, k, m) = G_calib(n, k, m)*fac;
                        else
                            countBlocked = countBlocked + 1;
                        end
                    end
                end
            end
        end
        fprintf('#LOS possibly blocked.. %d\n',countBlocked);
        
    case 3
        % This is the revised version based on calibration 2. Instead of
        % simply adding up the estimated tag phase offset (with magnitude),
        % which is equivalent of doing a weighted average, we do algebraic
        % average, without any weighting.
        scale = 1e-3;
        G_w_s = scale.*G_w_s;
        G_wo_s = scale.*G_wo_s;
        CableClutter = zeros(TagNum, RecvNum, FreqNum);
        CableClutter(:, 1, :) = 1;
        for m = 1:FreqNum
            kn = 2*pi*freq(m)/c;
            for k = 1:RecvNum
                for n = 1:TagNum
                    distTagRecv1 = norm(tagPosition(n, :) - rxPosition(1, :));
                    distTagRecv = norm(tagPosition(n, :) - rxPosition(k, :));
                    RelPhs = -kn*(distTagRecv - distTagRecv1);
                    % comment: why not 2 times the distance?
                    RelCompx = cos(RelPhs) + 1j*sin(RelPhs);
%                     RelCompx = exp(1i*RelPhs);
                    RelRawData = G_wo_s(n, k, m)*G_wo_s(n, 1, m)';
                    CableClutter(n, k, m) = RelCompx*RelRawData';
                    if abs(CableClutter(n, k, m)) > 0
                        CableClutter(n, k, m) = CableClutter(n, k, m)./abs(CableClutter(n, k, m));
                    end
                end
            end
        end
        
        % Calculate the phase offset introduced by each tag at each
        % frequency. Here we're calculating the tag loss averaging on all
        % the receiving antenna.
%                 TagCal = ones(FreqNum, TagNum);
        TagCal = zeros(FreqNum, TagNum);
        for m = 1:FreqNum
            kn = 2*pi*freq(m)/c;
            for n = 1:TagNum
                sumComp = 0;
                for k = 1:RecvNum
                    distTagRecv = norm(tagPosition(n, :) - rxPosition(k, :));
                    RelPhs = -kn*distTagRecv;
                    % comment: why not 2 times the distance?
                    RelCompx = cos(RelPhs) + 1j*sin(RelPhs);
%                     RelCompx = exp(1i*RelPhs);
                    RelRawData = G_wo_s(n, k, m)*CableClutter(n, k, m);
                    
                    % Normalize the complex number representing tag phase
                    % offset. This is the different part from calibration
                    % 2.
                    if abs(RelRawData) > 0
                        RelRawData = RelRawData/abs(RelRawData);
                    end
                    
                    sumComp = sumComp + RelCompx*RelRawData';
                end
                if abs(sumComp) > 0
                    TagCal(m, n) = sumComp/abs(sumComp);
                else
                    TagCal(m, n) = 1;
                end
            end
        end
        
        countBlocked = 0;
        for n = 1:TagNum
            for k = 1:RecvNum
                G_w_abs = abs(G_w_s(n, k, :));
                G_wo_abs = abs(G_wo_s(n, k, :));
                isLosBlocked = 1;
                for m = 1:FreqNum
                    if G_w_abs(m)*G_wo_abs(m) > 0
                        if G_w_abs(m)/G_wo_abs(m) < rssiDiffTh
                            isLosBLocked = 1*isLosBlocked;
                        else
                            isLosBlocked = 0*isLosBlocked;
                        end
                    end
                end
                
                for m = 1:FreqNum
                    if G_w_abs(m)*G_wo_abs(m) > 0
                        if isLosBlocked == 0
                            G_calib(n, k, m) = G_w_s(n, k, m) - G_wo_s(n, k, m);
                            fac = CableClutter(n, k, m)*TagCal(m, n);
                            G_calib(n, k, m) = G_calib(n, k, m)*fac;
                        else
                            countBlocked = countBlocked + 1;
                        end
                    end
                end
            end
        end
        fprintf('#LOS possibly blocked.. %d\n',countBlocked);
        
    case 4
        % Same as calibration 2, except including the part to evaluate the
        % tag phase offset for different receivers, instead of doing the
        % average.
        scale = 1e-3;
        G_w_s = scale.*G_w_s;
        G_wo_s = scale.*G_wo_s;
        CableClutter = zeros(TagNum, RecvNum, FreqNum);
        CableClutter(:, 1, :) = 1;
        for m = 1:FreqNum
            kn = 2*pi*freq(m)/c;
            for k = 1:RecvNum
                for n = 1:TagNum
                    distTagRecv1 = norm(tagPosition(n, :) - rxPosition(1, :));
                    distTagRecv = norm(tagPosition(n, :) - rxPosition(k, :));
                    RelPhs = -kn*(distTagRecv - distTagRecv1);
                    % comment: why not 2 times the distance?
                    RelCompx = cos(RelPhs) + 1j*sin(RelPhs);
%                     RelCompx = exp(1i*RelPhs);
                    RelRawData = G_wo_s(n, k, m)*G_wo_s(n, 1, m)';
%                     CableClutter(n, k, m) = RelCompx*RelRawData';
                    CableClutter(n, k, m) = 1283578;
                    if abs(CableClutter(n, k, m)) > 0
                        CableClutter(n, k, m) = CableClutter(n, k, m)./abs(CableClutter(n, k, m));
                    end
                end
            end
        end
        
        % Calculate the phase offset introduced by each tag at each
        % frequency. Here we're calculating the tag loss averaging on all
        % the receiving antenna.
%                 TagCal = ones(FreqNum, TagNum, RecvNum);
        TagCal = zeros(FreqNum, TagNum, RecvNum);
        for m = 1:FreqNum
            kn = 2*pi*freq(m)/c;
            for n = 1:TagNum
                for k = 1:RecvNum
                    distTagRecv = norm(tagPosition(n, :) - rxPosition(k, :));
                    RelPhs = -kn*distTagRecv;
                    % comment: why not 2 times the distance?
                    RelCompx = cos(RelPhs) + 1j*sin(RelPhs);
%                     RelCompx = exp(1i*RelPhs);
                    RelRawData = G_wo_s(n, k, m)*CableClutter(n, k, m);
                    TagCal(m, n, k) = RelCompx*RelRawData';
                    if abs(TagCal(m, n, k)) > 0
                        TagCal(m, n, k) = TagCal(m, n, k)/abs(TagCal(m, n, k));
                    else
                        TagCal(m, n, k) = 1;
                    end
                end
            end
        end
        
        countBlocked = 0;
        for n = 1:TagNum
            for k = 1:RecvNum
                G_w_abs = abs(G_w_s(n, k, :));
                G_wo_abs = abs(G_wo_s(n, k, :));
                isLosBlocked = 1;
                for m = 1:FreqNum
                    if G_w_abs(m)*G_wo_abs(m) > 0
                        if G_w_abs(m)/G_wo_abs(m) < rssiDiffTh
                            isLosBLocked = 1*isLosBlocked;
                        else
                            isLosBlocked = 0*isLosBlocked;
                        end
                    end
                end
                
                for m = 1:FreqNum
                    if G_w_abs(m)*G_wo_abs(m) > 0
                        if isLosBlocked == 0
                            G_calib(n, k, m) = G_w_s(n, k, m) - G_wo_s(n, k, m);
%                             fac = CableClutter(n, k, m)*TagCal(m, n, k);
                            fac = exp(-1i*(2*pi*freq(m)/c)*norm(tagPosition(n, :) - rxPosition(k, :)))*(G_wo_s(n, k, m)'/abs(G_wo_s(n, k, m)));
                            G_calib(n, k, m) = G_calib(n, k, m)*fac;
                        else
                            countBlocked = countBlocked + 1;
                        end
                    end
                end
            end
        end
        fprintf('#LOS possibly blocked.. %d\n',countBlocked);
    case 5
        % This is a different calibration from 2, 3, and 4. RF cable,
        % clutter, and tag circuitry are lumped into one variable to
        % represent the phase offset introduced by these sources. We expect
        % that by directly calculating the RF cable and clutter loss part,
        % instead of calculating the relative values (like in calibration
        % 2, 3, or 4), we are able to improve the results more.
%         scale = 1e-3;
%         G_w_s = scale.*G_w_s;
%         G_wo_s = scale.*G_wo_s;
        
        NonIdealCal = zeros(TagNum, RecvNum, FreqNum);
        for m = 1:FreqNum
            kn = 2*pi*freq(m)/c;
            for k = 1:RecvNum
                for n = 1:TagNum
                    distTagRecv = norm(tagPosition(n, :) - rxPosition(k, :));
                    CalPhs = -kn*distTagRecv;
                    CalCompx = cos(CalPhs) + 1j*sin(CalPhs);
%                     CalCompx = exp(1i*CalPhs);
                    NonIdealCal(n, k, m) = CalCompx*G_wo_s(n, k, m)';
                    if abs(NonIdealCal(n, k, m)) > 0
                        NonIdealCal(n, k, m) = NonIdealCal(n, k, m)/abs(NonIdealCal(n, k, m));
                    end
                    % We HAVE TO do normalization of the undesired part of
                    % the link, the "NonIdealCal" variable. If you don't
                    % normalize it, the results will be bad.
                end
            end
        end
        
        % Background subtraction. (We could lump this into the previous for
        % loop, but for ease of understanding, we will separate the
        % previous for loop, which is in charge of getting rid of the RF
        % cable, clutter and tag circuitry loss, and this one.)
        countBlocked = 0;
        for n = 1:TagNum
            for k = 1:RecvNum
                G_w_abs = abs(G_w_s(n, k, :));
                G_wo_abs = abs(G_wo_s(n, k, :));
%                 isLosBlocked = 0;
                isLosBlocked = 1;
                for m = 1:FreqNum
                    if G_w_abs(m)*G_wo_abs(m) > 0
                        if G_w_abs(m)/G_wo_abs(m) < rssiDiffTh
                            isLosBLocked = 1*isLosBlocked;
                        else
                            isLosBlocked = 0*isLosBlocked;
                        end
                    end
                end
                
                for m = 1:FreqNum
                    if G_w_abs(m)*G_wo_abs(m) > 0
                        if isLosBlocked == 0
                            % Equivalent implementation.
%                             G_calib(n, k, m) = G_w_s(n, k, m) - G_wo_s(n, k, m);
%                             G_calib(n, k, m) = G_calib(n, k, m)*NonIdealCal(n, k, m);

                            % Equivalent implementation.
                            G_w_s(n, k, m) = G_w_s(n, k, m)*NonIdealCal(n, k, m);
                            G_wo_s(n, k, m) = G_wo_s(n, k, m)*NonIdealCal(n, k, m);
                            G_calib(n, k, m) = G_w_s(n, k, m) - G_wo_s(n, k, m);
                            
                            % Equivalent implementation. (But this
                            % implementation is too complicated in
                            % expression.)
%                             G_w_s(n, k, m) = G_w_s(n, k, m)*(G_wo_s(n, k, m)'/abs(G_wo_s(n, k, m)))*exp(1i*(-(2*pi*freq(m)/c)*norm(tagPosition(n, :) - rxPosition(k, :))));
%                             G_wo_s(n, k, m) = G_wo_s(n, k, m)*(G_wo_s(n, k, m)'/abs(G_wo_s(n, k, m)))*exp(1i*(-(2*pi*freq(m)/c)*norm(tagPosition(n, :) - rxPosition(k, :))));
%                             G_wo_s(n, k, m) = (abs(G_wo_s(n, k, m))^2/abs(G_wo_s(n, k, m)))*exp(1i*(-(2*pi*freq(m)/c)*norm(tagPosition(n, :) - rxPosition(k, :))));
%                             G_calib(n, k, m) = G_w_s(n, k, m) - G_wo_s(n, k, m);
                            % This above implementation is equivalent to
                            % the previous two because NonIdealCal(n, m, k)
                            % = (G_wo_s(n, k, m)'/abs(G_wo_s(n, k, m)))*
                            % exp(-1i*(2*pi*freq(m)/c)*norm(tagPosition
                            % - rxPosition(k, :))). Therefore they are
                            % exactly the same implementations.

                        else
                            countBlocked = countBlocked + 1;
                        end
                    end
                end
            end
        end
        fprintf('#LOS possibly blocked.. %d\n',countBlocked);
        % Questions: The variable "NonIdealCal" represents everything else
        % except the LoS propagation delay. 1) Why the propagation delay is
        % the distance from tag to receiver, not twice the distance from
        % tag to receiver? 2) Why doesn't it matter if we normalize
        % "NonIdealCal" or not, but it does matter if we normalize "G_wo_s"
        % or not?
        
        % Thoughts: We could first do background subtraction. Then we are
        % theoretically left with only the signal reflected by the object,
        % as well as RF cable loss, background clutter associated with the
        % link of object reflection, and the tag circuitry loss (all
        % complex numbers). There could be several ways to get rid of all
        % of these undesired parts and make only the object reflection
        % remain, including differential method, and non-differential
        % methods. Let me think about this now.
    case 6
        % This is a different calibration from 2, 3, and 4. RF cable,
        % clutter, and tag circuitry are lumped into one variable to
        % represent the phase offset introduced by these sources. We expect
        % that by directly calculating the RF cable and clutter loss part,
        % instead of calculating the relative values (like in calibration
        % 2, 3, or 4), we are able to improve the results more.
%         scale = 1e-3;
%         G_w_s = scale.*G_w_s;
%         G_wo_s = scale.*G_wo_s;
        
        NonIdealCal = zeros(TagNum, RecvNum, FreqNum);
        for m = 1:FreqNum
            kn = 2*pi*freq(m)/c;
            for k = 1:RecvNum
                for n = 1:TagNum
                    distTagRecv = norm(tagPosition(n, :) - rxPosition(k, :));
                    CalPhs = -kn*distTagRecv;
                    CalCompx = cos(CalPhs) + 1j*sin(CalPhs);
%                     CalCompx = exp(1i*CalPhs);
                    NonIdealCal(n, k, m) = CalCompx*G_wo_s(n, k, m)';
                    if abs(NonIdealCal(n, k, m)) > 0
                        NonIdealCal(n, k, m) = NonIdealCal(n, k, m)/abs(NonIdealCal(n, k, m));
                    end
                    % We HAVE TO do normalization of the undesired part of
                    % the link, the "NonIdealCal" variable. If you don't
                    % normalize it, the results will be bad.
                end
            end
        end
        
        % Background subtraction. (We could lump this into the previous for
        % loop, but for ease of understanding, we will separate the
        % previous for loop, which is in charge of getting rid of the RF
        % cable, clutter and tag circuitry loss, and this one.)
        countBlocked = 0;
        for n = 1:TagNum
            for k = 1:RecvNum
                G_w_abs = abs(G_w_s(n, k, :));
                G_wo_abs = abs(G_wo_s(n, k, :));
                isLosBlocked = 0;
%                 isLosBlocked = 1;
%                 for m = 1:FreqNum
%                     if G_w_abs(m)*G_wo_abs(m) > 0
%                         if G_w_abs(m)/G_wo_abs(m) < rssiDiffTh
%                             isLosBLocked = 1*isLosBlocked;
%                         else
%                             isLosBlocked = 0*isLosBlocked;
%                         end
%                     end
%                 end
                
                for m = 1:FreqNum
                    if G_w_abs(m)*G_wo_abs(m) > 0
                        if isLosBlocked == 0
                            % Equivalent implementation.
%                             G_calib(n, k, m) = G_w_s(n, k, m) - G_wo_s(n, k, m);
%                             G_calib(n, k, m) = G_calib(n, k, m)*NonIdealCal(n, k, m);

                            % Equivalent implementation.
                            G_w_s(n, k, m) = G_w_s(n, k, m)*NonIdealCal(n, k, m);
                            G_wo_s(n, k, m) = G_wo_s(n, k, m)*NonIdealCal(n, k, m);
                            G_calib(n, k, m) = G_w_s(n, k, m) - G_wo_s(n, k, m);
                            
                            % Equivalent implementation. (But this
                            % implementation is too complicated in
                            % expression.)
%                             G_w_s(n, k, m) = G_w_s(n, k, m)*(G_wo_s(n, k, m)'/abs(G_wo_s(n, k, m)))*exp(1i*(-(2*pi*freq(m)/c)*norm(tagPosition(n, :) - rxPosition(k, :))));
%                             G_wo_s(n, k, m) = G_wo_s(n, k, m)*(G_wo_s(n, k, m)'/abs(G_wo_s(n, k, m)))*exp(1i*(-(2*pi*freq(m)/c)*norm(tagPosition(n, :) - rxPosition(k, :))));
%                             G_wo_s(n, k, m) = (abs(G_wo_s(n, k, m))^2/abs(G_wo_s(n, k, m)))*exp(1i*(-(2*pi*freq(m)/c)*norm(tagPosition(n, :) - rxPosition(k, :))));
%                             G_calib(n, k, m) = G_w_s(n, k, m) - G_wo_s(n, k, m);
                            % This above implementation is equivalent to
                            % the previous two because NonIdealCal(n, m, k)
                            % = (G_wo_s(n, k, m)'/abs(G_wo_s(n, k, m)))*
                            % exp(-1i*(2*pi*freq(m)/c)*norm(tagPosition
                            % - rxPosition(k, :))). Therefore they are
                            % exactly the same implementations.

                        else
                            countBlocked = countBlocked + 1;
                        end
                    end
                end
            end
        end
        fprintf('#LOS possibly blocked.. %d\n',countBlocked);
    case 7
        % We combine Calibration 2 with differential receiving, in hope for
        % eliminating initial phase associated with the received data, with
        % RF cable, clutter and tag circuitry loss subtracted.
    case 8
        % We combine Calibration 5 with differential receiving, in hope for
        % eliminating initial phase associated with the received data, with
        % RF cable, clutter and tag circuitry together subtracted.
    otherwise
        fprintf('Wrong calibration option selection, select a valid method. \n');
end
b = G_calib(:);

%% Reconstruction.
% The voxel coordinates.
x_v=0:0.05:1.5;
y_v=0:0.05:1.5;
z_v=0:0.08:2.4;

NxVoxel = length(x_v);
NyVoxel = length(y_v);
NzVoxel = length(z_v);

% Calculate all the distances needed.
VoxelCoord = combvec(x_v, y_v, z_v);
p_xVoxel = VoxelCoord(1, :);
p_yVoxel = VoxelCoord(2, :);
p_zVoxel = VoxelCoord(3, :);

p_tagx = zeros(TagNum, NxVoxel*NyVoxel*NzVoxel);
p_tagy = zeros(TagNum, NxVoxel*NyVoxel*NzVoxel);
p_tagz = zeros(TagNum, NxVoxel*NyVoxel*NzVoxel);
p_tagx(:, 1) = tagPosition(:, 1);
p_tagy(:, 1) = tagPosition(:, 2);
p_tagz(:, 1) = tagPosition(:, 3);
p_tagx = repmat(p_tagx(:, 1), [1, NxVoxel*NyVoxel*NzVoxel]);
p_tagy = repmat(p_tagy(:, 1), [1, NxVoxel*NyVoxel*NzVoxel]);
p_tagz = repmat(p_tagz(:, 1), [1, NxVoxel*NyVoxel*NzVoxel]);

p_recvx = zeros(RecvNum, NxVoxel*NyVoxel*NzVoxel);
p_recvy = zeros(RecvNum, NxVoxel*NyVoxel*NzVoxel);
p_recvz = zeros(RecvNum, NxVoxel*NyVoxel*NzVoxel);
p_recvx(:, 1) = rxPosition(:, 1);
p_recvy(:, 1) = rxPosition(:, 2);
p_recvz(:, 1) = rxPosition(:, 3);
p_recvx = repmat(p_recvx(:, 1), [1, NxVoxel*NyVoxel*NzVoxel]);
p_recvy = repmat(p_recvy(:, 1), [1, NxVoxel*NyVoxel*NzVoxel]);
p_recvz = repmat(p_recvz(:, 1), [1, NxVoxel*NyVoxel*NzVoxel]);

DistTagVoxel = sqrt((p_tagx - repmat(p_xVoxel, [TagNum, 1])).^2 ...
    +(p_tagy - repmat(p_yVoxel, [TagNum, 1])).^2+(p_tagz - repmat(p_zVoxel, [TagNum, 1])).^2);

DistRecvVoxel = sqrt((p_recvx - repmat(p_xVoxel, [RecvNum, 1])).^2 ...
    +(p_recvy - repmat(p_yVoxel, [RecvNum, 1])).^2+(p_recvz - repmat(p_zVoxel, [RecvNum, 1])).^2);

DistTemp = repmat(DistTagVoxel, RecvNum, 1) + repelem(DistRecvVoxel, TagNum, 1);
DistUplink = repmat(DistTemp, length(freq), 1);

% Calculate the frequency constant.
FreqConst = (repelem(-1j*2*pi*freq/c, TagNum*RecvNum)).';
Freq_ratio = (repelem(freq/min(freq), TagNum*RecvNum).^2).';
% A = Freq_ratio.*exp(FreqConst.*DistUplink);
A = exp(FreqConst.*DistUplink);

% Reconstruct the object reflectivity for each voxel.
tic;
imgComplex = A'*b;
tComp = toc;

%% Post-Processing.
% Apply threshold for reconstructed reflectivity intensity.
imgComplexAbs = abs(imgComplex).^2;
% imgComplexAbs = abs(imgComplex);
ReconsDistMax = max(imgComplexAbs);
ReconsDistMin = min(imgComplexAbs);
ReconsDistNorm = (imgComplexAbs - ReconsDistMin)/(ReconsDistMax - ReconsDistMin);

Threshold = ReconsDistMax*0.6;
% image_3D = imgComplexAbs;
% image_3D(imgComplexAbs < Threshold)=nan;
% image_3D = reshape(image_3D, [length(x_v), length(y_v), length(z_v)]);
imgComplexAbs = reshape(imgComplexAbs, [length(x_v), length(y_v), length(z_v)]);
imgComplexAbs(imgComplexAbs < Threshold) = 0;
%ReconsDistNorm(ReconsDistNorm <0.1 ) = 0;

% imgComplexAbs(imgComplexAbs >= Threshold) = 1;



%% Plotting.
% Convert the relative/normalized brightness of the reconstructed
% reflectivity vector into a 3D matrix.
ReconsDistNorm = reshape(ReconsDistNorm, [length(x_v), length(y_v), length(z_v)]);

% Plot the relative/normalized brightness of the reconstructed
% reflectivity, in a 3D grid.
[X_V, Y_V, Z_V] = meshgrid(x_v, y_v, z_v);
ReconsDistNorm = permute(ReconsDistNorm, [2 1 3]);
[gx1,gy1,gz1] = gradient(ReconsDistNorm);
 g1=sqrt(gx1.^2 +gy1.^2 +gz1.^2);
 g1Max=max(g1(:));

Threshold = g1Max*0.7;
g1(g1 < Threshold) = 0;
%g1(g1 > Threshold) = 1;
%h = slice(X_V, Y_V, Z_V, ReconsDistNorm,x_v,y_v,z_v);
h = slice(X_V, Y_V, Z_V, g1,x_v,y_v,z_v);
% imgComplexAbs(imgComplexAbs == 0) = NaN;
% h = slice(X_V, Y_V, Z_V, imgComplexAbs,x_v,y_v,z_v);
xlabel('x / m','FontSize',14);
ylabel('y / m','FontSize',14);
zlabel('z / m','FontSize',14);
xlim([x_v(1), x_v(length(x_v))]);
ylim([y_v(1), y_v(length(y_v))]);
zlim([z_v(1), z_v(length(z_v))]);

set(h, 'EdgeColor','none',...
    'FaceColor','interp',...
    'FaceAlpha','interp');
alpha('color');
a = alphamap('rampup',256);
imgThresh = 150;
a(1:imgThresh) = 0;
alphamap(a);
set(gca, 'fontweight', 'bold');

hold on
scatter3(2.4, 1.8, 1.2, 5, 'bo', 'LineWidth', 5);

% Clustering.
roomSize = [x_v(1), x_v(length(x_v)); y_v(1), y_v(length(y_v)); z_v(1), z_v(length(z_v))];
voxelSize = [x_v(2)-x_v(1); y_v(2)-y_v(1); z_v(2)-z_v(1)];
[cDistribution, clusters,centroidDist] = i4block_components(g1, roomSize, voxelSize);

fprintf('Initial cluster number = %d\n',length(clusters.centroid));
for i = 1:size(clusters.centroid,1)
    fprintf('Clusters centroid: [%3.2f, %3.2f, %3.2f] with element number %d\n',clusters.centroid(i,:),clusters.elemNum(i));
end
fprintf('\n');
opts.distTh = 0.2; % distance threshold, clusters with centers closer than this will be combined
opts.XYdistTh = 0.2;
opts.elemNumTh = 0.61; % clusters with element number less than 60% of the maximum will be rejected
opts.minHeightRatio = 0.6; % Minimum height ratio compared to largest object, exact ht depends on voxel size etc.
clusterOut = clusterProcess(clusters,opts);

centroid = clusterOut.centroid;
elemNum = clusterOut.elemNum;
for i = 1:size(centroid,1)
    fprintf('Clusters centroid: [%3.2f, %3.2f, %3.2f] with element number %d\n',centroid(i,:),elemNum(i));
end
%{
% Plot the tags and receiver antennas.
figure;
x0=400;
y0=200;
width=900;
height=700;
set(gcf,'position',[x0,y0,width,height]);

scatter3(tagPosition(:, 1), tagPosition(:, 2), tagPosition(:, 3), 'r*');
hold on;
scatter3(rxPosition(:, 1), rxPosition(:, 2), rxPosition(:, 3), 'b');
% axis 'equal';

grid on
xt = 0:0.5:x_v(length(x_v));
yt = 0:0.5:y_v(length(y_v));
zt = 0:0.5:z_v(length(z_v));
set(gca,'xtick',xt, 'xticklabel', 100*xt);
set(gca,'ytick',yt, 'yticklabel', 100*yt);
set(gca,'ztick',zt, 'zticklabel', 100*zt);

xlim([-0.4 4.2]);
ylim([-0.4 4.2]);
zlim([0.8 2.6]);
legend('Tags','Receivers');
set(gca, 'FontSize', 15);
xlabel('x coordinate (cm)');
ylabel('y coordinate (cm)');
zlabel('z coordinate (cm)');
%}

% % Plot the image.
% figure;
% x0=400;
% y0=200;
% width=900;
% height=700;
% set(gcf,'position',[x0,y0,width,height]);
% 
% [X_V, Y_V, Z_V] = meshgrid(x_v, y_v, z_v);
% image_3D = permute(image_3D, [2 1 3]);
% h = slice(X_V, Y_V, Z_V, image_3D,[],[],z_v);
% % axis 'equal';
% 
% grid on
% xt = 0:0.5:x_v(length(x_v));
% yt = 0:0.5:y_v(length(y_v));
% zt = 0:0.5:z_v(length(z_v));
% set(gca,'xtick',xt, 'xticklabel', 100*xt);
% set(gca,'ytick',yt, 'yticklabel', 100*yt);
% set(gca,'ztick',zt, 'zticklabel', 100*zt);
% 
% set(h, 'EdgeColor','none', 'FaceColor','interp')
% set(gca, 'FontSize', 15);
% xlabel('x coordinate (cm)');
% ylabel('y coordinate (cm)');
% zlabel('z coordinate (cm)');

