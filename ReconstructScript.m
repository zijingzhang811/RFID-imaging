

% Parameters.
[tagPosition, rxPosition, freq] = tag_antenna_positions3D_func();
TagNum = size(tagPosition, 1);
RecvNum = size(rxPosition, 1);
FreqNum = length(freq);
c = physconst('LightSpeed'); 

%% Data Collection.

% Threshold for standard deviation of phase.
phStdTh = 15; % In degrees. A very high value means no rejection.
% fprintf('Using phase standard deviation threshold %d\n', phStdTh);
rssiDiffTh = .5;    % Set the threshold of RSSI. This variable is useless is you set "RJT" equal to 0.
RJT = 1;    % 1 means you threshold the RSSI, and other values means you don't threshold the RSSI.

% Specify data directory and name.
dirName = 'F:\RFID imaging\data 0129\';
% fileName = 'wo_1224';
opts.calibType = 1;

% Load the raw data.
load([dirName, 'data_wo02','.mat']); % Looking at 2 minute cases
% data_wo = [chindexlist, tagindexlist, antennalist, rssiimpinjlist, rssiimpinjlist_d, phasedeglist];

% Generate the data matrices.
PhaseDataCollect1 = cell(TagNum, RecvNum, FreqNum);
PhaseCollectStd1 = zeros(TagNum, RecvNum, FreqNum);
PhaseCollectMean1 = zeros(TagNum, RecvNum, FreqNum);
RSSIDataCollect1 = cell(TagNum, RecvNum, FreqNum);
RSSICollectStd1 = zeros(TagNum, RecvNum, FreqNum);
RSSICollectMean1 = zeros(TagNum, RecvNum, FreqNum);
CompNumDataCollect1 = cell(TagNum, RecvNum, FreqNum);
CompNumCollectMean1 = zeros(TagNum, RecvNum, FreqNum);
for j = 1:RecvNum
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
                PhaseDataCollect1{i, j, k} = phase_ttt;
                PhaseCollectStd1(i, j, k) = std(phase_ttt);
                RSSIDataCollect1{i, j, k} = rssi_ttt;
                RSSICollectStd1(i, j, k) = std(rssi_ttt);
                CompNumDataCollect1{i, j, k} = rssi_ttt.*cos(deg2rad(phase_ttt)) + sqrt(-1).*rssi_ttt.*sin(deg2rad(phase_ttt));
                if PhaseCollectStd1(i, j, k) > phStdTh
                    % First completely reject the channel using phase
                    % standard deviation threshold.
                    PhaseCollectMean1(i, j, k) = NaN;
                    RSSICollectMean1(i, j, k) = NaN;
                    CompNumCollectMean1(i, j, k) = NaN;
                else
                    PhaseCollectMean1(i, j, k) = mean(phase_ttt);
                    RSSICollectMean1(i, j, k) = mean(rssi_ttt);
                    CompNumCollectMean1(i, j, k) = mean(CompNumDataCollect1{i, j, k});
                end
            else
                % Simply there's no data received for this channel.
                PhaseDataCollect1{i, j, k} = NaN;
                PhaseCollectStd1(i, j, k) = NaN;
                PhaseCollectMean1(i, j, k) = NaN;
                RSSIDataCollect1{i, j, k} = NaN;
                RSSICollectStd1(i, j, k) = NaN;
                RSSICollectMean1(i, j, k) = NaN;
                CompNumDataCollect1{i, j, k} = NaN;
                CompNumCollectMean1(i, j, k) = NaN;
            end                
        end
    end
end
clear chindexlist tagindexlist antennalist rssiimpinjlist rssiimpinjlist_d phasedeglist

% Load the raw data.
load([dirName, 'data_mbox02','.mat']); 
% data_w = [chindexlist, tagindexlist, antennalist, rssiimpinjlist, rssiimpinjlist_d, phasedeglist];

% Generate the data matrices.
PhaseDataCollect2 = cell(TagNum, RecvNum, FreqNum);
PhaseCollectStd2 = zeros(TagNum, RecvNum, FreqNum);
PhaseCollectMean2 = zeros(TagNum, RecvNum, FreqNum);
RSSIDataCollect2 = cell(TagNum, RecvNum, FreqNum);
RSSICollectStd2 = zeros(TagNum, RecvNum, FreqNum);
RSSICollectMean2 = zeros(TagNum, RecvNum, FreqNum);
CompNumDataCollect2 = cell(TagNum, RecvNum, FreqNum);
CompNumCollectMean2 = zeros(TagNum, RecvNum, FreqNum);
for j = 1:RecvNum
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
                PhaseDataCollect2{i, j, k} = phase_ttt;
                PhaseCollectStd2(i, j, k) = std(phase_ttt);
                RSSIDataCollect2{i, j, k} = rssi_ttt;
                RSSICollectStd2(i, j, k) = std(rssi_ttt);
                CompNumDataCollect2{i, j, k} = rssi_ttt.*cos(deg2rad(phase_ttt)) + sqrt(-1).*rssi_ttt.*sin(deg2rad(phase_ttt));
                if PhaseCollectStd2(i, j, k) > phStdTh
                    % First completely reject the channel using phase
                    % standard deviation threshold.
                    PhaseCollectMean2(i, j, k) = NaN;
                    RSSICollectMean2(i, j, k) = NaN;
                    CompNumCollectMean2(i, j, k) = NaN;
                else
                    PhaseCollectMean2(i, j, k) = mean(phase_ttt);
                    RSSICollectMean2(i, j, k) = mean(rssi_ttt);
                    CompNumCollectMean2(i, j, k) = mean(CompNumDataCollect2{i, j, k});
                end
            else
                % Simply there's no data received for this channel.
                PhaseDataCollect2{i, j, k} = NaN;
                PhaseCollectStd2(i, j, k) = NaN;
                PhaseCollectMean2(i, j, k) = NaN;
                RSSIDataCollect2{i, j, k} = NaN;
                RSSICollectStd2(i, j, k) = NaN;
                RSSICollectMean2(i, j, k) = NaN;
                CompNumDataCollect2{i, j, k} = NaN;
                CompNumCollectMean2(i, j, k) = NaN;
            end                
        end
    end
end
clear chindexlist tagindexlist antennalist rssiimpinjlist rssiimpinjlist_d phasedeglist


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
x_v=0:0.03:1.5;
y_v=0:0.03:1.5;
z_v=0:0.03:1.5;

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

Threshold = ReconsDistMax*0.8;
% image_3D = imgComplexAbs;
% image_3D(imgComplexAbs < Threshold)=nan;
% image_3D = reshape(image_3D, [length(x_v), length(y_v), length(z_v)]);
imgComplexAbs = reshape(imgComplexAbs, [length(x_v), length(y_v), length(z_v)]);
imgComplexAbs(imgComplexAbs < Threshold) = 0;
imgComplexAbs(imgComplexAbs >= Threshold) = 1;

% Clustering.
roomSize = [x_v(1), x_v(length(x_v)); y_v(1), y_v(length(y_v)); z_v(1), z_v(length(z_v))];
voxelSize = [x_v(2)-x_v(1); y_v(2)-y_v(1); z_v(2)-z_v(1)];
[cDistribution, clusters,~] = i4block_components(imgComplexAbs, roomSize, voxelSize);

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


%% Plotting.
% Convert the relative/normalized brightness of the reconstructed
% reflectivity vector into a 3D matrix.
ReconsDistNorm = reshape(ReconsDistNorm, [length(x_v), length(y_v), length(z_v)]);

% Plot the relative/normalized brightness of the reconstructed
% reflectivity, in a 3D grid.
[X_V, Y_V, Z_V] = meshgrid(x_v, y_v, z_v);
ReconsDistNorm = permute(ReconsDistNorm, [2 1 3]);
ReconsDistNorm(ReconsDistNorm<0.8)=0;
imgBrightnessf = csaps({x_v,y_v,z_v},ReconsDistNorm,0.99999);
imgSpline = fnval(imgBrightnessf,{x_v,y_v,z_v}) ;
[gx1,gy1,gz1] = gradient(imgSpline);
 g1=sqrt(gx1.^2 +gy1.^2 +gz1.^2);

figure
h = slice(X_V, Y_V, Z_V,ReconsDistNorm ,x_v,y_v,z_v);
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
imgThresh = 200;
a(1:imgThresh) = 0;
alphamap(a);
set(gca, 'fontweight', 'bold');

hold on
scatter3(2.4, 1.8, 1.2, 5, 'bo', 'LineWidth', 5);

figure
h = slice(X_V, Y_V, Z_V,g1 ,x_v,y_v,z_v);
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
 g1(g1<0.2)=0;
  g1(g1>0.2)=1;
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

%Plot the tags and receiver antennas.
figure;
x0=400;
y0=200;
width=900;
height=700;
set(gcf,'position',[x0,y0,width,height]);

scatter3(tagPosition(:, 1), tagPosition(:, 2), tagPosition(:, 3), 'r*');
hold on;
scatter3(rxPosition(:, 1), rxPosition(:, 2), rxPosition(:, 3), 'b');
axis 'equal';

grid on
xt = 0:0.5:x_v(length(x_v));
yt = 0:0.5:y_v(length(y_v));
zt = 0:0.5:z_v(length(z_v));
set(gca,'xtick',xt, 'xticklabel', 100*xt);
set(gca,'ytick',yt, 'yticklabel', 100*yt);
set(gca,'ztick',zt, 'zticklabel', 100*zt);

xlim([0 1.8]);
ylim([0 1.8]);
zlim([0 1.8]);
legend('Tags','Receivers');
set(gca, 'FontSize', 15);
xlabel('x coordinate (cm)');
ylabel('y coordinate (cm)');
zlabel('z coordinate (cm)');


% Plot the image.
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

