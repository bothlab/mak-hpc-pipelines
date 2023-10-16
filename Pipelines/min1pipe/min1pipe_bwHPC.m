function [file_name_to_save, filename_reg] = min1pipe_bwHPC(Fsi, spatialr, se, ismc, flag, fname_video_raw, dir_results)
    %%% wrapper for MIN1PIPE, called by a Python helper script %%%

    MIN1PIPE_ROOT_DIR = '<dataroot>/HPC/Tools/MIN1PIPE-v2+git/';
    cd(MIN1PIPE_ROOT_DIR);
    addpath(genpath(MIN1PIPE_ROOT_DIR))
    run([MIN1PIPE_ROOT_DIR, '/utilities/cvx/cvx_startup.m']);

    %% configure paths %%
    min1pipe_init([MIN1PIPE_ROOT_DIR, '/']);

    %% initialize parameters %%
    if nargin < 1 || isempty(se)
        defpar = default_parameters;
        se = defpar.neuron_size;
    end

    if nargin < 2 || isempty(ismc)
        ismc = true;
    end

    if nargin < 3 || isempty(flag)
        flag = 1;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%% parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% user defined parameters %%%                                     %%%
    Params.Fsi = Fsi;                                                   %%%
    Params.Fsi_new = Params.Fsi;                                        %%%
    Params.spatialr = spatialr;                                         %%%
    Params.neuron_size = se; %%% half neuron size; 9 for Inscopix and 5 %%%
                            %%% for UCLA, with 0.5 spatialr separately  %%%
                                                                        %%%
    %%% fixed parameters (change not recommanded) %%%                   %%%
    Params.anidenoise_iter = 4;                   %%% denoise iteration %%%
    Params.anidenoise_dt = 1/7;                   %%% denoise step size %%%
    Params.anidenoise_kappa = 0.5;       %%% denoise gradient threshold %%%
    Params.anidenoise_opt = 1;                %%% denoise kernel choice %%%
    Params.anidenoise_ispara = 1;             %%% if parallel (denoise) %%%   
    Params.bg_remove_ispara = 1;    %%% if parallel (backgrond removal) %%%
    Params.mc_scl = 0.004;      %%% movement correction threshold scale %%%
    Params.mc_sigma_x = 5;  %%% movement correction spatial uncertainty %%%
    Params.mc_sigma_f = 10;    %%% movement correction fluid reg weight %%%
    Params.mc_sigma_d = 1; %%% movement correction diffusion reg weight %%%
    Params.pix_select_sigthres = 0.8;     %%% seeds select signal level %%%
    Params.pix_select_corrthres = 0.6; %%% merge correlation threshold1 %%%
    Params.refine_roi_ispara = 1;          %%% if parallel (refine roi) %%%
    Params.merge_roi_corrthres = 0.9;  %%% merge correlation threshold2 %%% 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    disp('User parameters:');
    disp(['Fsi: ', num2str(Fsi)]);
    disp(['spatialr: ', num2str(spatialr)]);
    disp(['neuron_size: ', num2str(se)]);
    disp(['ismc: ', num2str(ismc)]);
    disp(['flag: ', num2str(flag)]);
    fprintf('\n');

    %% cluster parallel setup for Matlab %%
    % create a local cluster object
    pc = parcluster('local');

    % get the number of dedicated cores from environment
    num_workers = str2double(getenv('SLURM_NTASKS'));
    if num_workers < 16
        num_workers = 16;
    end
    if num_workers > 16
        disp('Limiting worker number to 16 to avoid running out of GPU memory.');
        num_workers = 16;
    end

    % explicitly set the JobStorageLocation to the tmp directory that is unique to each cluster job (and is on local, fast scratch)
    parpool_tmpdir = [getenv('TMPDIR'), '/.matlab/local_cluster_jobs/slurm_jobID_', getenv('SLURM_JOB_ID')];

    mkdir(parpool_tmpdir);
    pc.JobStorageLocation = parpool_tmpdir;

    % start the parallel pool
    parpool(pc, num_workers, 'SpmdEnabled', false);

    %% run %%
    
    hpipe = tic;
    overwrite_flag = true; % always override existing data
    if overwrite_flag
        %% load data from raw video file %%
        defpar = default_parameters;
        stype = parse_type(defpar.ttype);
        m = matfile(fname_video_raw);

        [pixh, pixw, nf] = size(m, 'frame_all');
        imaxn = zeros(pixh, pixw);
        iminf = zeros(pixh, pixw);

        nsize = pixh * pixw * nf * stype; %%% size of single %%%
        nbatch = batch_compute(nsize);
        ebatch = ceil(nf / nbatch);
        idbatch = [1: ebatch: nf, nf + 1];

        imaxn = max(cat(3, max(m.frame_all, [], 3), imaxn), [], 3);
        iminf = min(cat(3, min(m.frame_all, [], 3), iminf), [], 3);

        imx1 = max(imaxn(:));
        imn1 = min(iminf(:));
        m = normalize_batch(fname_video_raw, 'frame_all', imx1, imn1, idbatch);
        save([dir_results, 'mp_supporting.mat'], 'imx1', 'imn1')

        clear nsize nbatch ebatch idbatch;
        %%

        %% neural enhancing batch version %%
        filename_reg = [dir_results, 'mp_reg.mat'];
        [m, imaxy, overwrite_flag] = neural_enhance(m, filename_reg, Params);

        %% neural enhancing postprocess %%
        m = noise_suppress(m, imaxy);

        %% get rough roi domain %%
        mask = dominant_patch(imaxy);

        %% frame register %%
        if ismc && overwrite_flag
            pixs = min(pixh, pixw);
            Params.mc_pixs = pixs;
            Fsi_new = Params.Fsi_new;
            scl = Params.neuron_size / (7 * pixs);
            sigma_x = Params.mc_sigma_x;
            sigma_f = Params.mc_sigma_f;
            sigma_d = Params.mc_sigma_d;
            se = Params.neuron_size;
            [m, corr_score, raw_score, scl] = frame_reg(m, imaxy, se, Fsi_new, pixs, scl, sigma_x, sigma_f, sigma_d);
            Params.mc_scl = scl; %%% update latest scl %%%

            file_name_to_save = [dir_results, 'mp_data_processed.mat'];
            if exist(file_name_to_save, 'file')
                delete(file_name_to_save)
            end
            save(file_name_to_save, 'corr_score', 'raw_score', '-v7.3');
        end

        time1 = toc(hpipe);

        %% select pixel %%
        if flag == 1
            sz = Params.neuron_size;
            Fsi_new = Params.Fsi_new;
            sigthres = Params.pix_select_sigthres;
            corrthres = Params.pix_select_corrthres;
            [roi, sig, bg, bgf, seeds, datasmth0, cutoff0, pkcutoff0] = pix_select(m, mask, sz, Fsi_new, sigthres, corrthres);
            time1 = toc(hpipe);
        else
            sz = Params.neuron_size;
            Fsi_new = Params.Fsi_new;
            [roi, sig, seeds, bg, bgf, datasmth0, cutoff0, pkcutoff0] = manual_seeds_select(m, Fsi_new, sz);
        end

        %% parameter init %%
        hpipe = tic;
        [P, options] = par_init(m);

        %% refine roi %%
        noise = P.sn;
        ispara = Params.refine_roi_ispara;
        [roirf, bgr, sigupdt, seedsupdt, datasmthf1, cutofff1, pkcutofff1] = refine_roi(m, sig, bgf, roi, seeds, noise, datasmth0, cutoff0, pkcutoff0, ispara);

        %% refine sig %%
        p = 0; %%% no ar model used %%%
        [sigrf, bgrf, Puse] = refine_sig(m, roirf, bgr, sigupdt, bgf, p, options);

        %% merge roi %%
        corrthres = Params.merge_roi_corrthres;
        [roimrg, sigmrg, seedsmrg, datasmthf2, cutofff2, pkcutofff2] = merge_roi(m, roirf, sigrf, seedsupdt, datasmthf1, cutofff1, pkcutofff1, corrthres);

        %% refine roi again %%
        Puse.p = 0;
        Puse.options = options;
        Puse.noise = noise;
        ispara = Params.refine_roi_ispara;
        [roifn, bgfn, sigupdt2, seedsfn] = refine_roi(m, sigmrg, bgrf, roimrg, seedsmrg, Puse.noise, datasmthf2, cutofff2, pkcutofff2, ispara);

        %% refine sig again for raw sig %%
        Puse.p = 0; %%% 0 ar model used %%%
        [sigfnr, ~, ~] = refine_sig(m, roifn, bgfn, sigupdt2, bgf, Puse.p, Puse.options);
        sigfnr = max(roifn, [], 1)' .* sigfnr;
        roifnr = roifn ./ max(roifn, [], 1);

        %% refine sig again %%
        Puse.p = 2; %%% 2nd ar model used %%%
        Puse.options.p = 2;
        [sigfn, bgffn, ~, spkfn] = refine_sig(m, roifn, bgfn, sigupdt2, bgf, Puse.p, Puse.options);
        sigfn = max(roifn, [], 1)' .* sigfn;
        roifn = roifn ./ max(roifn, [], 1);
%             dff = sigfn ./ mean(sigfn, 2);
        dff = sigfn ./ mean(bgffn);

        %% save data %%
        stype = parse_type(class(m.reg(1, 1, 1)));
        nsize = pixh * pixw * nf * stype; %%% size of single %%%
        nbatch = batch_compute(nsize);
        ebatch = ceil(nf / nbatch);
        idbatch = [1: ebatch: nf, nf + 1];
        nbatch = length(idbatch) - 1;
        imax = zeros(pixh, pixw);
        for j = 1: nbatch
            tmp = m.reg(1: pixh, 1: pixw, idbatch(j): idbatch(j + 1) - 1);
            imax = max(cat(3, max(tmp, [], 3), imax), [], 3);
        end

        file_name_to_save = [dir_results, 'mp_data_processed.mat'];
        if exist(file_name_to_save, 'file')
            if ismc
                load(file_name_to_save, 'raw_score', 'corr_score')
            end
            delete(file_name_to_save)
        end

        if ismc
            save(file_name_to_save, 'roifn', 'sigfn', 'dff', 'seedsfn', 'spkfn', 'bgfn', 'bgffn', 'roifnr', 'sigfnr', 'imax', 'pixh', 'pixw', 'corr_score', 'raw_score', 'Params', '-v7.3');
        else
            save(file_name_to_save, 'roifn', 'sigfn', 'dff', 'seedsfn', 'spkfn', 'bgfn', 'bgffn', 'roifnr', 'sigfnr', 'imax', 'pixh', 'pixw', 'Params', '-v7.3');
        end

        save(file_name_to_save, 'imaxn', 'imaxy', '-append');
        time2 = toc(hpipe);
        disp(['Done all, total time: ', num2str(time1 + time2), ' seconds'])
    else
        filename_reg = [dir_results, 'mp_reg.mat'];
        file_name_to_save = filecur;

        time2 = toc(hpipe);
        disp(['Done all, total time: ', num2str(time2), ' seconds'])
    end
end


function min1pipe_init(mp_root)
% parse path, and install cvx if not
%   Jinghao Lu, 11/10/2017

    %%% check if on path %%%
    pathCell = regexp(path, pathsep, 'split');
    if ispc  % Windows is not case-sensitive
        onPath = any(strcmpi(mp_root(1: end - 1), pathCell)); %%% get rid of filesep %%%
    else
        onPath = any(strcmp(mp_root(1: end - 1), pathCell));
    end

    %%% set path and setup cvx if not on path %%%
    if ~onPath
        pathall = genpath(mp_root);
        addpath(pathall)
        cvx_dir = [mp_root, 'utilities'];
        if ~exist([cvx_dir, filesep, 'cvx'], 'dir')
            if ispc
                cvxl = 'http://web.cvxr.com/cvx/cvx-w64.zip';
            elseif isunix
                cvxl = 'http://web.cvxr.com/cvx/cvx-a64.zip';
            elseif ismac
                cvxl = 'http://web.cvxr.com/cvx/cvx-maci64.zip';
            end
            disp('Downloading CVX');
            unzip(cvxl, cvx_dir);
        end
        pathcvx = [cvx_dir, filesep, 'cvx', filesep, 'cvx_setup.m'];
        run(pathcvx)
    end
end
