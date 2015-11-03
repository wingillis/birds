function [ data, response_detected, voltage ] = stimulate(stim, hardware, detrend_param, handles)

% stim = [ current_uA, halftime_us, interpulse_s, n_repetitions,
% repetition_Hz, active_electrodes, polarities ]

% hardware = [ ni.session, tdt, hardware.plexon.id, stim.plexon_monitor_electrode ]

global currently_reconfiguring;
global scriptdir;
global axes3yy;


if currently_reconfiguring
    disp('Still reconfiguring the hardware... please wait (about 3 seconds, usually)...');
    return;
end

set(handles.currentcurrent, 'String', sigfig(stim.current_uA, 2));
set(handles.halftime, 'String', sprintf('%.1f', stim.halftime_us));

      

% A is amplitude, W is width, Delay is interphase delay.

StimParamPos.A1 = stim.current_uA;
StimParamPos.A2 = -stim.current_uA;
StimParamPos.W1 = stim.halftime_us;
StimParamPos.W2 = stim.halftime_us;
StimParamPos.Delay = stim.interpulse_s * 1e6;

StimParamNeg.A1 = -stim.current_uA;
StimParamNeg.A2 = stim.current_uA;
StimParamNeg.W1 = stim.halftime_us;
StimParamNeg.W2 = stim.halftime_us;
StimParamNeg.Delay = stim.interpulse_s * 1e6;


NullPattern.W1 = 0;
NullPattern.W2 = 0;
NullPattern.A1 = 0;
NullPattern.A2 = 0;
NullPattern.Delay = 0;

filenamePos = strrep(strcat(scriptdir, '/stimPos.pat'), '/', filesep);
filenameNeg = strrep(strcat(scriptdir, '/stimNeg.pat'), '/', filesep);
plexon_write_rectangular_pulse_file(filenamePos, StimParamPos);
plexon_write_rectangular_pulse_file(filenameNeg, StimParamNeg);


try

    % If no stim.plexon_monitor_electrode is selected, just fail silently and let the user figure
    % out what's going on :)
    if stim.plexon_monitor_electrode > 0 & stim.plexon_monitor_electrode <= 16
        err = PS_SetMonitorChannel(hardware.plexon.id, stim.plexon_monitor_electrode);
        if err
            ME = MException('plexon:monitor', 'Could not set monitor channel to %d', stim.plexon_monitor_electrode);
            throw(ME);
        end
    end
    
    %disp('stimulating on channels:');
    %stim
    for channel = find(stim.active_electrodes)
        err = PS_SetPatternType(hardware.plexon.id, channel, 1);
        if err
            ME = MException('plexon:pattern', 'Could not set pattern type on channel %d', channel);
            throw(ME);
        end

        if stim.negativefirst(channel)
            err = PS_LoadArbPattern(hardware.plexon.id, channel, filenameNeg);
        else
            err = PS_LoadArbPattern(hardware.plexon.id, channel, filenamePos);
        end
        if err
            ME = MException('plexon:pattern', 'Could not set pattern parameters on channel %d, because %d (%s)', ...
                channel, err, PS_GetExtendedErrorInfo(err));
            throw(ME);
        end
        
        if true
            np = PS_GetNPointsArbPattern(hardware.plexon.id, channel);
            pat = [];
            pat(1,:) = PS_GetArbPatternPointsX(hardware.plexon.id, channel);
            pat(2,:) = PS_GetArbPatternPointsY(hardware.plexon.id, channel);
            pat = [[0; 0] pat [pat(1,end); 0]]; % Add zeros for cleaner look
            if ~isempty(axes3yy) & isvalid(axes3yy)
                hold(axes3yy(2), 'on');
                plot(axes3yy(2), pat(1,:)/1e6, pat(2,:)/1e3, 'g');
                hold(axes3yy(2), 'off');
                legend(axes3, 'Voltage', 'Current', 'Next i');
            end
        end
        
        switch hardware.stim_trigger
            case 'master8'
                err = PS_SetRepetitions(hardware.plexon.id, channel, 1);
            case 'arduino'
                err = PS_SetRepetitions(hardware.plexon.id, channel, 1);
            case 'ni'
                err = PS_SetRepetitions(hardware.plexon.id, channel, stim.n_repetitions);
            case 'plexon'
                err = PS_SetRepetitions(hardware.plexon.id, channel, stim.n_repetitions);
            otherwise
                disp(sprintf('You must set a valid value for hardware.stim_trigger. ''%s'' is invalid.', hardware.stim_trigger));
        end       
        if err
            ME = MException('plexon:pattern', 'Could not set repetitions on channel %d', channel);
            throw(ME);
        end
        
        err = PS_SetRate(hardware.plexon.id, channel, stim.repetition_Hz);
        if err
            ME = MException('plexon:pattern', 'Could not set repetition rate on channel %d', channel);
            throw(ME);
        end

        [v, err] = PS_IsWaveformBalanced(hardware.plexon.id, channel);
        if err
            ME = MException('plexon:stimulate', 'Bad parameter for stimbox %d channel %d', hardware.plexon.id, channel);
            throw(ME);
        end
        if ~v
            ME = MException('plexon:stimulate:unbalanced', 'Waveform is not balanced for stimbox %d channel %d', hardware.plexon.id, channel);
            throw(ME);
        end


        err = PS_LoadChannel(hardware.plexon.id, channel);
        if err
            ME = MException('plexon:stimulate', 'Could not stimulate on box %d channel %d: %s', hardware.plexon.id, channel, PS_GetExtendedErrorInfo(err));    
            throw(ME);
        end
    end
    
    switch hardware.stim_trigger
        case 'master8'
            err = PS_SetTriggerMode(hardware.plexon.id, 1);
        case 'arduino'
            err = PS_SetTriggerMode(hardware.plexon.id, 1);
        case 'ni'
            err = PS_SetTriggerMode(hardware.plexon.id, 1);
        case 'plexon'
            err = PS_SetTriggerMode(hardware.plexon.id, 0);
    end
    if err
        ME = MException('plexon:trigger', 'Could not set trigger mode on channel %d', channel);
        throw(ME);
    end
                
    
    if isfield(hardware, 'tdt') && ~isempty(hardware.tdt)
        hardware.tdt.device.SetTagVal('mon_gain', round(hardware.tdt.audio_monitor_gain/5));
    end
    

    switch hardware.stim_trigger
        case 'master8'
            [ event.Data, event.TimeStamps ] = hardware.ni.session.startForeground;
        case 'arduino'
            [ event.Data, event.TimeStamps ] = hardware.ni.session.startForeground;
        case 'ni'
            [ event.Data, event.TimeStamps ] = hardware.ni.session.startForeground;
        case 'plexon'
            hardware.ni.session.startBackground;
            err = PS_StartStimAllChannels(hardware.plexon.id);
            if err
                hardware.ni.session.stop;
                ME = MException('plexon:stimulate', 'Could not stimulate on box %d: %s', hardware.plexon.id, PS_GetExtendedErrorInfo(err));
                throw(ME);
            end
            hardware.ni.session.wait;  % This callback needs to be interruptible!  Apparently it is??
    end
    
    
    if isfield(hardware, 'tdt') && ~isempty(hardware.tdt)
        hardware.tdt.device.SetTagVal('mon_gain', hardware.tdt.audio_monitor_gain);
    end
    
    [ data, response_detected, voltage ] = organise_data(stim, hardware, detrend_param, event, handles);


catch ME
    
    errordlg(ME.message, 'Error', 'modal');
    disp(sprintf('Caught the error %s (%s).  Shutting down...', ME.identifier, ME.message));
    report = getReport(ME)
    rethrow(ME);
end







% Write out a file that defines an arbitrary rectangular pulse for the
% Plexon. This gives sub-uA control, rather than the 1-uA control given by
% their default rectangular pulse interface.
function plexon_write_rectangular_pulse_file(filename, StimParam);
fid = fopen(filename, 'w');
fprintf(fid, 'variable\n');
fprintf(fid, '%d\n%d\n', round(StimParam.A1*1000), round(StimParam.W1));
if StimParam.Delay
    fprintf(fid, '%d\n%d\n', 0, round(StimParam.Delay));
end
fprintf(fid, '%d\n%d\n', round(StimParam.A2*1000), round(StimParam.W2));
fclose(fid);





