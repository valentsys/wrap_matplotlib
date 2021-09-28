"""
Copyright (c) 2021 Vladislav Ivanov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.patches import FancyArrowPatch, Circle, Ellipse
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticky
import matplotlib.colors as mcolors

import numpy as np

maxfontsize= 22

#HELPERS
def preserve_eps_text():  
  plt.rcParams['svg.fonttype'] = 'none' #preseve text as text


def extr_param(name, cont, i= -1):
  param_= None
  if(name in cont):
    param_= cont[name][i] if (isinstance(cont[name], list) and i>= 0) else cont[name]
  return param_

def setup_grid(plt, majorcolor= '#404040', minorcolor= '#595959'):
  plt.minorticks_on()
  plt.grid(b=False, color= majorcolor, linestyle='-', linewidth=0.8)
  plt.grid(b=False, which='minor', color= minorcolor, linestyle='-', alpha=0.2)
  #plt.locator_params(axis='x', nbins=15)

def format_axis_E(axisobj):
  axisobj.set_major_formatter(ticky.FormatStrFormatter('%.1E'))

def D_frm(x, pos):
  return "%d" % (x)

def E_frm(x, pos):
  return "%.1E" % (x)

def d3_frm(x, pos):
  return "%d" % int(x*1e3)

def m3_frm(x, pos):
  return "%.1f" % (x*1e3)

def mn20_frm(x, pos):
  return "%.1f" % (x*1e-20)

def m3_2f_frm(x, pos):
  return "%.2f" % (x*1e3)

def m0_3f_frm(x, pos):
  return "%.3f" % (x)

def m0_frm(x, pos):
  return "%.1f" % x

def m02_frm(x, pos):
  return "%.2f" % x

def format_axis_func(axisobj, frmfunc):
  axisobj.set_major_formatter(ticky.FuncFormatter(frmfunc))

def disable_xaxis(subplt, lblbottom= False):
    subplt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom= lblbottom
  )

#frame snapshot plots
"""
def make_colormap(colormap_filename):
  rgb_colours = extract_colmap_colors(colormap_filename)
  return ListedColormap(rgb_colours[::-1])
"""

#xaxis format [data, label, lims, frmfunc, inv]
#xscale can be: linear, log, symlog, logit
def x_(data, label, xlow= None, xhigh= None, frmfunc= None, xticks= None, xscale= "linear"):
  return { "d": data, "lab": label, "lims": [xlow, xhigh], "frmfunc": frmfunc, "inv": False, "xticks": xticks, "scal": xscale }

#yaxis format [data, label, legend, lims, markers, style, frmfunc, col, lthk]
def y_(data, label, legend= None, lthk= 2, color= None, ylow= None, yhigh= None, noticks= False, styl= None, marker= None, frmfunc= None, yscale= "linear"):
  return { "d": data, "lab": label, "leg": legend, "lims": [ylow, yhigh], "noticks": noticks, "markers": marker, "style": styl if styl is not None else "-", "frmfunc": frmfunc, "col": color, "lthk": lthk, "scal": yscale }

def colorz_(data, label= None, cmapname= "rainbow", zlow= None, zhigh= None, frmfunc= None, docontour= False):
  return { "d": data, "lab": label, "lims": [zlow, zhigh], "cmapname": cmapname, "frmfunc": frmfunc, "docont": docontour }

def arrowz_(datax, datay, label= None, qsett= None, zlow= None, zhigh= None, frmfunc= None):
  return { "d": (datax, datay), "lab": label, "sett": qsett, "frmfunc": frmfunc }

def quiverkeyp_(X= 0.46, Y= 0.9, U= 5, label= 'scale= 5 m/s', labelpos= 'E'):
  return {
    "X": X, "Y": Y, "U": U, "label": label, "labelpos": labelpos
  }

def quiverp_(
    width= 1.6e-2, scale= 23.2, linewidth= 0.8, minshaft= 1.2
  , edgecolor= "black", facecolor= "white", qksett= None
):
  return {
      "width": width, "scale": scale, "linewidth": linewidth
    , "minshaft": minshaft, "edgecolor": edgecolor, "facecolor": facecolor, "qksett": qksett
  }


def gen_fig(N= 1, M= 1, dims= [12, 10]):#N is for i, M is for j
  fig, axs = plt.subplots(ncols= N, nrows= M, figsize=(dims[0], dims[1])) # single figure setup)
  return [fig, axs]

def fig_ax_(figobj, i= 0, j= 0):
  fig_= figobj[0]
  axs_= figobj[1]
  col_= axs_[i] if(hasattr(axs_, "size")) else axs_
  elem_= col_[j] if(hasattr(col_, "size")) else col_
  return [fig_, elem_]

def pargs_(xdata, ydata, zdata= None, maxfntsize= maxfontsize, title= None, legpos= None, aspect= "auto", nogrid= False):

  return [
    [xdata, ydata, zdata]
  , { "max_fontsize": maxfntsize, "title": title, "legpos": legpos, "aspect": aspect, "nogrid": nogrid}
  ]

def add_text(ax, x, y, str_, fntsize= maxfontsize*0.9):
  props = dict(boxstyle='round', facecolor='white', alpha=0.9)
  plt.text(x, y, str_, horizontalalignment='center', verticalalignment='center', transform = ax[1].transAxes, fontsize= fntsize, bbox=props)
  

def fig_plot(figobj_, filename= None, mtop= None, mright= None, mbottom= None, mleft= None, wspace= None, hspace= None):

  fig_= figobj_[0]

  #actual plot
  #plt.tight_layout()
  plt.subplots_adjust(top= mtop, left= mleft, right= mright, bottom= mbottom, hspace= hspace, wspace= wspace)

  if(filename is None):
    plt.show()
  else:
    plt.savefig(filename, bbox_inches='tight')
    fig_.clear()

  plt.close(fig_)


def add_errorbars(figax, xdata, ydata, yerrdata, capsize_= 10, elinewidth_= 3, ecolor_= 'black', barswidth_= 2):
  (_, caps, _) = figax[1].errorbar(
    xdata, ydata, yerr= yerrdata, ls='none', capsize=capsize_, elinewidth=elinewidth_, ecolor= ecolor_)

  for cap in caps:
    cap.set_markeredgewidth(barswidth_)

def add_legend(figobj_, maxfntsize, legloc= "upper center", ncols= 1, figpos= [0.5,0.95]):
  fig_= figobj_[0]
  
  fig_.legend(
      loc= legloc, fontsize= maxfntsize*0.6
    , ncol= ncols
    , fancybox=True
    , bbox_transform=fig_.transFigure
    , bbox_to_anchor=(figpos[0], figpos[1])
  )

def add_title(figobj_, title, maxfntsize):
  fig_= figobj_[0]
  fig_.suptitle(title, fontsize= maxfntsize)

def add_plot(args, fig):

  data= args[0]
  figprops= args[1]

  fig_= fig[0]
  ax= fig[1]

  #figprops
  fsize_max= figprops["max_fontsize"]

  #set title
  if ("title" in figprops):
    ax.set_title(figprops["title"], size= fsize_max)

  #fig rescaling
  if ("aspect" in figprops):
    ax.set_aspect(figprops["aspect"])

  #fig_.subplots_adjust(right=0.80)

  #extract data
  xvar= data[0]
  yvars= data[1]

  #plot data
  axi= 0
  ##axref= [0]*len(yvars)
  axref= ax
  axs= [None]*len(yvars)

  plts_= [None]*len(yvars)

  def extr_param(name, cont, i= 0):
    param_= None
    if(name in cont):
      param_= cont[name][i] if (isinstance(cont[name], list)) else cont[name]
    return param_

  for yvar in yvars:

    xvar_= xvar["d"][axi] if isinstance(xvar["d"], list) and len(xvar["d"])> axi else xvar["d"]

    if(isinstance(yvar["d"], list)):
      k_= 0
      plts_[axi]= []
      for yvar__ in yvar["d"]:
        xvar__= xvar_[k_] if isinstance(xvar_, list) and len(xvar_)> k_ else xvar_
        col_= extr_param("col", yvar, k_)
        styl_= extr_param("style", yvar, k_)
        mark_= extr_param("markers", yvar, k_)
        lthk_= extr_param("lthk", yvar, k_)
        leg_= extr_param("leg", yvar, k_)

        plts_[axi].append(
            axref.plot(
              xvar__
            , yvar__
            , label= leg_
            , linewidth= lthk_
            , color= col_
            , linestyle= styl_
            , marker= mark_
          )
        )
        k_= k_+1
    else:
      col_= extr_param("col", yvar)
      styl_= extr_param("style", yvar)
      mark_= extr_param("markers", yvar)
      lthk_= extr_param("lthk", yvar)
      leg_= extr_param("leg", yvar)

      plts_[axi] = axref.plot(
          xvar_
        , yvar["d"]
        , label= leg_
        , linewidth= lthk_
        , color= col_
        , linestyle= styl_
        , marker= mark_
      )

    #set y scale
    if("scal" in yvar and yvar["scal"] is not None):
      axref.set_yscale(yvar["scal"])

    #set y label & limit
    if(axi> 0):
      axref.set_ylabel(yvar["lab"], size= fsize_max*0.9, rotation=0)#, rotation=0 if the label needs to be horizontal
    else:
      axref.set_ylabel(yvar["lab"], size= fsize_max*0.9)

    if("lims" in yvar and yvar["lims"] is not None):
      axref.set_ylim(yvar["lims"])


    #format y axis
    axref.tick_params(axis="both", labelsize= fsize_max*0.8)
    if("frmfunc" in yvar and yvar["frmfunc"] is not None):
      format_axis_func(axref.yaxis, yvar["frmfunc"])

    #set x scale what?
    if("scal" in xvar and xvar["scal"] is not None):
      axref.set_xscale(xvar["scal"])

    if(axi> 0):
      disable_xaxis(axref)
      axref.patch.set_visible(False)
      axref.yaxis.set_label_coords(1.0 + 0.23*(axi-1), 1.06)
      axref.spines["right"].set_position(("axes", 1.0 + 0.23*(axi-1)))
      #make_patch_spines_invisible(ax)
      #setup axis grid
      #setup_grid(axref)

      #rem axis
      axs[axi]= axref

      #clone axis
      axi= axi+1
      if(axi== len(yvars)):
        break
      axref= ax.twinx()

  if("nogrid" not in figprops or figprops["nogrid"] == False):
    setup_grid(ax)

  #set x label & limit
  ax.set_xlabel(xvar["lab"], size= fsize_max*0.9)
  if("lims" in xvar and xvar["lims"] is not None):
    ax.set_xlim(xvar["lims"])

  if("xticks" in xvar and xvar["xticks"] is not None):
    plt.xticks(ticks= xvar["xticks"][0], labels= xvar["xticks"][1], fontsize= fsize_max*0.8)

  #format x axis
  if("frmfunc" in xvar and xvar["frmfunc"] is not None):
    format_axis_func(ax.xaxis, xvar["frmfunc"])

  if ("legpos" in figprops and figprops["legpos"] is not None):
    legpos_= figprops["legpos"]
    if(isinstance(legpos_, list) and len(legpos_) == len(axs)):
      for ax_ in axs:
        j_= 0

        if(isinstance(plts_[axi], list)):
          lns= []
          labs= []*len(plts_)
          for plt_ in plts_:
            if(isinstance(plt_, list) and len(plt_)> 1):
              for subplt_ in plt_:
                lns= lns+ subplt_
            else:
              lns= lns+ plt_

          labs= [l.get_label() for l in lns]
          ax_.legend(lns, labs,
              loc= legpos_[j_], fontsize= fsize_max*0.7
            , fancybox=True
          )
        else:
          ax_.legend(
              loc= legpos_[j_], fontsize= fsize_max*0.7
            , fancybox=True
          )

        j_= j_+1

    else:
      #aggregate plots for common legend
      lns= []
      labs= []*len(plts_)
      for plt_ in plts_:
        if(isinstance(plt_, list) and len(plt_)> 1):
          for subplt_ in plt_:
            lns= lns+ subplt_
        else:
          lns= lns+ plt_

      labs= [l.get_label() for l in lns]

      ax.legend(lns, labs,
          loc= figprops["legpos"], fontsize= fsize_max*0.7
        , fancybox=True
      )

#2D plots

def add_colorsurface_plot(args, fig):

  data= args[0]
  figprops= args[1]

  fig_= fig[0]
  ax= fig[1]

  #figprops
  fsize_max= extr_param("max_fontsize", figprops)
  if(fsize_max is None): fsize_max= maxfontsize

  #set title
  title_= extr_param("title", figprops)
  if (title_ is not None): ax.set_title(figprops["title"], size= fsize_max*0.9, y=1.02)

  #fig rescaling
  aspect_= extr_param("aspect", figprops)
  if ("aspect" in figprops): ax.set_aspect(figprops["aspect"])

  #fig_.subplots_adjust(right=0.80)

  #extract data
  xvar= data[0]#[0] if(isinstance(data[0], list) and len(data[0])>= 1) else data[0]
  yvar= data[1]#[0] if(isinstance(data[1], list) and len(data[1])>= 1) else data[1]
  zvar= data[2][0] if(isinstance(data[2], list) and len(data[2])>= 1) else data[2]

  col_= extr_param("col", yvar)
  leg_= extr_param("leg", yvar)

  cmapname_= extr_param("cmapname", zvar)

  xx_= xvar["d"]
  yy_= yvar["d"]
  Z_= zvar["d"]

  xxp_= xx_[0] if(isinstance(xx_, list) and len(xx_)>= 1) else xx_
  yyp_= yy_[0] if(isinstance(yy_, list) and len(yy_)>= 1) else yy_

  #plotting surface
  #normalize = mcolors.clors.Normalize(vmin= lims_[0], vmax= lims_[1])
  docont_= extr_param("docont", zvar)
  if(docont_):
    
    # Label every other level using strings
    cs= ax.contourf(xxp_, yyp_, Z_, cmap= cmapname_, levels= 7, rasterized= True)#, norm=normalize)
    cs2= ax.contour(xxp_, yyp_, Z_, levels= 7)#, norm=normalize)

    CLS = plt.clabel(cs2, inline=1, fontsize=10)

    # now CLS is a list of the labels, we have to find offending ones
    thresh = 0.05  # ratio in x/y range in border to discard

    # get limits if they're automatic
    xmin,xmax,ymin,ymax = plt.axis()
    Dx = xmax-xmin
    Dy = ymax-ymin

    # check which labels are near a border
    keep_labels = []
    j_=0
    for label in CLS:
      lx,ly = label.get_position()
      #if xmin+thresh*Dx<lx<xmax-thresh*Dx and ymin+thresh*Dy<ly<ymax-thresh*Dy:
          # inlier, redraw it later
      keep_labels.append((0,ly))
      j_= j_+1

    # delete the original lines, redraw manually the labels we want to keep
    # this will leave unlabelled full contour lines instead of overlapping labels

    for cline in cs2.collections:
        cline.remove()
    for label in CLS:
        label.remove()

    cs2= ax.contour(xxp_, yyp_, Z_, levels= 7, colors='k')#, norm=normalize)
    CLS = plt.clabel(cs2, inline=1, fontsize= fsize_max*0.7, manual=keep_labels, fmt='%.2fT', colors= "black")
  else:
    cs= ax.contourf(xxp_, yyp_, Z_, cmap= cmapname_, rasterized= True)#, norm=normalize)
  #cs= ax.pcolormesh(xxp_, yyp_, Z_, cmap= cmapname_, shading= "gouraud", rasterized= True)
  
  #setting the colorbar

  lims_= extr_param("lims", zvar)
  if(lims_ is not None):
    cs.set_clim(vmin= lims_[0], vmax= lims_[1])

  lab_= extr_param("lab", zvar)
  zfrmfunc_= extr_param("frmfunc", zvar)
  if(lab_ is not None):
    cbar_= fig_.colorbar(cs, ax= ax, fraction= 0.07, pad= 0.12)
    cbar_.ax.tick_params(labelsize= fsize_max*0.8)
    cbar_.ax.set_ylabel(None, size= fsize_max*0.8)
    if(zfrmfunc_ is not None):
      format_axis_func(cbar_.ax.yaxis, zfrmfunc_)
    cbar_.ax.yaxis.get_offset_text().set_fontsize(fsize_max*0.65)
      #cbar_.ax.set_xlim(lims_[0], lims_[1])
    #cbar_.ax.set_ylabel(lab_, size= fsize_max*0.6)
    #cbar_= fig.colorbar(cs, ax=ax, fraction= 0.09, pad= 0.05) <- demo

  #set y scale
  yscal_= extr_param("scal", yvar)
  if(yscal_ is not None): ax.set_yscale(yscal_)

  #set y label & limit
  ax.set_ylabel(yvar["lab"], size= fsize_max*0.8)

  ylims_= extr_param("lims", yvar)
  if(ylims_ is not None): ax.set_ylim(ylims_)

  #format y axis
  ax.tick_params(axis="both", labelsize= fsize_max*0.8)
  yfrmfunc_= extr_param("frmfunc", yvar)
  if(yfrmfunc_ is not None): format_axis_func(ax.yaxis, yfrmfunc_)

  noticks_= extr_param("noticks", yvar)
  if(noticks_ is True):
    ax.set(yticklabels=[])
    ax.tick_params(left=False)

  if("nogrid" not in figprops or figprops["nogrid"] == False):
    setup_grid(ax)

  #set x label & limit
  ax.set_xlabel(xvar["lab"], size= fsize_max*0.8)

  xlims_= extr_param("lims", xvar)
  if(xlims_ is not None): ax.set_xlim(xlims_)

  xticks_= extr_param("xticks", xvar)
  if(xticks_ is not None): plt.xticks(ticks= xticks_[0], labels= xticks_[1], fontsize= fsize_max*0.8)

  #format x axis
  xfrmfunc_= extr_param("frmfunc", xvar)
  if(xfrmfunc_ is not None): format_axis_func(ax.xaxis, xfrmfunc_)

  #add quiver plot
  xxa_= xx_[1] if(isinstance(xx_, list) and len(xx_)>= 2) else None
  yya_= yy_[1] if(isinstance(yy_, list) and len(yy_)>= 2) else None
  zzvar= data[2][1] if(isinstance(data[2], list) and len(data[2])>= 2) else None

  q_= None
  if(xxa_ is not None and yya_ is not None and zzvar is not None):
    ZU_= zzvar["d"][0]
    ZV_= zzvar["d"][1]

    qsett= extr_param("sett", zzvar)

    #plotting arrows
    q_= ax.quiver(
      xxa_, yya_, ZU_, ZV_
      , width= qsett["width"], scale= qsett["scale"]
      , linewidth= qsett["linewidth"], minshaft= qsett["minshaft"]
      , edgecolor= qsett["edgecolor"], facecolor= qsett["facecolor"], rasterized= True
    )
    qksett= extr_param("qksett", qsett)
    if(qksett is not None):
      ax.quiverkey(
          q_, X= qksett["X"], Y= qksett["Y"]
        , U= qksett["U"], label= qksett["label"]
        , labelpos= qksett["labelpos"], coordinates= "figure"
        , fontproperties= { "size": fsize_max*0.8 }
      )

  return cs, q_

def plot_colorbar(cs_, ax_, lab_= None, fract_= 0.07, pad_= 0.12, lab= None, low_= None, high_= None, ax_s= [0.95, 0.3, 0.02, 0.5], fsize_max= maxfontsize, frmfunc= None):

  #setting the colorbar
  #cbar_= ax_[0].colorbar(cs_, ax= ax_[1], fraction= fract_, pad= pad_)
  cbar_ax = ax_[0].add_axes(ax_s)
  cbar_= ax_[0].colorbar(cs_, cax=cbar_ax, fraction= fract_, pad= pad_)
  if(low_ is not None): cs_.set_clim(vmin= low_)
  if(high_ is not None): cs_.set_clim(vmax= high_)

  cbar_.ax.tick_params(labelsize= fsize_max*0.8)
  cbar_.ax.set_ylabel(None, size= fsize_max*0.8)
  cbar_.ax.yaxis.get_offset_text().set_fontsize(fsize_max*0.65)

  if(frmfunc is not None):
    format_axis_func(cbar_.ax.yaxis, frmfunc)

  if(lab_ is not None): cbar_.ax.set_ylabel(lab_, size= fsize_max*0.8)
  #cbar_= fig.colorbar(cs, ax=ax, fraction= 0.09, pad= 0.05) <- demo