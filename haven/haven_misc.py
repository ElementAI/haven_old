import pandas as pd
import pylab as plt 


def merge_pdfs(fname_list, output_name="output.pdf"):
    from PyPDF2 import PdfFileReader, PdfFileMerger
    import PyPDF2
    import img2pdf

    from pdfrw import PdfReader, PdfWriter

    writer = PdfWriter()
    for inpfn in fname_list:
        decrypt_pdf(inpfn, output_name)
        writer.addpages(PdfReader(output_name).pages)
    writer.write(output_name)
    

def decrypt_pdf(input_name, output_name):
    import pikepdf

    pdf = pikepdf.open(input_name)
    pdf.save(output_name)

def images_to_pdf(fname_list, output_name="output.pdf"):
    from fpdf import FPDF
    pdf = FPDF()
    # imagelist is the list with all image filenames
    for image in fname_list:
        pdf.add_page()
        pdf.image(image,x=0,y=0,w=610,h=297)
    pdf.output(output_name, "F")

def get_bar_chart(score_list, label_list, sep, ylabel, fontsize, title, width, legend_flag=False, figsize=(20,10)):
    """
    label_list = ['LCFCN', 'WSLM', 'Glance-ram', 'Glance']
    score_list = [2.12, 3.10, 3.38, 4.03]
    get_bar_chart(score_list=score_list, label_list=label_list, ylabel='MAE', width=0.35, sep=.5, fontsize=fontsize,
                                    title='Mall')
    """
    fig, ax = plt.subplots(figsize = figsize, dpi=200)
    ind = np.arange(len(score_list))*sep
    plt.title(title, fontsize=fontsize+4)
    width = width   
    for i in range(len(ind)):
        index = ind[i]
        value = score_list[i]

        rects = ax.bar([index+width], [value], width=width, label=label_list[i])
    
    minimum, maximum = ax.get_ylim()
    y = .05 * (maximum - minimum)

    for i in range(len(ind)):
        index = ind[i]
        value = score_list[i]
    
        ax.text(x=index + width - .025 , y = y, s=f"{value}" , fontdict=dict(fontsize=fontsize+2), color='white')

    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    ax.set_ylabel(ylabel, fontsize=fontsize+2)
    ax.grid(True)

    if legend_flag:
        plt.legend(**{'fontsize':fontsize, "loc":2, "bbox_to_anchor":(1.05,1),
                                              'borderaxespad':0., "ncol":1})
    else:
        ax.set_xticks(ind + width)
        ax.set_xticklabels(label_list)
    plt.tight_layout()
    
    plt.show()
    