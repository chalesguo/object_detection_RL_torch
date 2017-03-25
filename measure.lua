local function averagePrecision(tp,fp,npos)
    fp = torch.cumsum(fp)
    tp = torch.cumsum(tp)
    local rec = tp/npos
    local prec = torch.cdiv(tp,(fp+tp+1e-16))
    local ap=torch.zeros(1)
    for t=0,1,0.1 do
        local tmp = prec[rec:ge(t)]
        local p =0
        if tmp:mDimension()>0 then
            p = torch.max(tmp)
        end
        if p < 1 then
            p = 0
        end
        ap = ap + p/11
    end
    return rec,prec,ap
end

local function xVOCap(rec,prec)
    local mrec = torch.cat(torch.zeros(1),rec):cat(torch.ones(1))
    local mpre = torch.cat(torch.zeros(1),prec):cat(torch.zeros(1))
    for i =mpre:size(1)-1,1,-1 do
        mpre[i] = math.max(mpre[i],mpre[i+1])
    end
    local indexA = torch.ByteTensor(mrec:size()):zero()
    local indexB = torch.ByteTensor(mrec:size()):zero()
    for ii = 2,mrec:size(1) do
        if mrec[ii-1]~=mrec[ii] then
            indexA[ii] = 1
            indexB[ii-1] = 1
        end
    end
    local ap = torch.sum((mrec[indexA]-mrec[indexB]):cmul(mpre[indexB]))
    return ap
end

local function evaluateTpFp(matches,gt)
    local iou,tp,fp,npos =0,0,0,0
    npos = #gt.rois
    for i,m in ipairs(matches) do
        local roi_m
        if m.p>0.9 then
            roi_m = m.random
        else
            goto continue
        end

        for igt,vgt in ipairs(gt) do
            iou = Rect.Iou(vgt.rect,roi.m)
            if iou >0.5 then
                if vgt.class_index == m.class then
                    tp = tp+1
                else
                fp = fp +1
                end
            end
        end
        ::continue::
    end
    return tp,fp,npos
end

function plot_training_progress(prefix, stats)
  local fn_p = string.format('%s/%sproposal_progress.png',opt.resultDir,prefix)
  local fn_d = string.format('%s/%sdetection_progress.png',opt.resultDir,prefix)
  gnuplot.pngfigure(fn_p)
  gnuplot.title('Traning progress over time (proposal)')

  local xs = torch.range(1, #stats.pcls)

  gnuplot.plot(
    { 'preg', xs, torch.Tensor(stats.preg), '-' },
    { 'pcls', xs, torch.Tensor(stats.pcls), '-' }
  )

  gnuplot.axis({ 0, #stats.pcls, 0, 10 })
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('loss')

  gnuplot.pngfigure(fn_d)
  gnuplot.title('Traning progress over time (detection)')

  gnuplot.plotflush()
  gnuplot.plot(
    { 'dreg', xs, torch.Tensor(stats.dreg), '-' },
    { 'dcls', xs, torch.Tensor(stats.dcls), '-' }
  )

  gnuplot.axis({ 0, #stats.pcls, 0, 10 })
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('loss')

  gnuplot.plotflush()
end