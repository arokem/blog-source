#!/usr/bin/env python
import os
import os.path as op
import time

import IPython.nbformat.current as nbformat
import IPython.nbconvert as nbconvert

start_py_high = '.. code-block:: python'

def add_newline(this_str):
    """ 
    Some kinda boilerplate
    """
    if not this_str.endswith('\n'):
        return this_str + '\n'
    else:
        return this_str
    

class CustomMarkdownExporter(nbconvert.MarkdownExporter):
    """
    My custom markdown converter. Uses some IPython machinery, but has some
    particular quirks


    For my idiosyncratic use, I want to return a string with the text that
    will be written into my output markdown file, as well as a bunch of
    filenames to where the figures ended up being saved. 
    """
    def __init__(self, config=None, extra_loaders=None, **kw):
        super(CustomMarkdownExporter, self).__init__(config=None,
                                                     extra_loaders=None,
                                                     **kw)

    def from_file(self, f):
        """
        
        """
        
        self._converted = super(CustomMarkdownExporter, self).from_file(f)
        f.seek(0)
        # This gives me the png graphics 
        self.pngs = self._converted[1]['outputs']
        # XXX What about non 
        nb = nbformat.read(f, 'ipynb')
        ws = nb['worksheets']
        self.cells = ws[0]['cells']    
    
    def from_filename(self, fname): 
        """
        """
        self.fname = fname
        self.basename = op.basename(self.fname).split('.ipynb')[0]
        return self.from_file(open(fname))    
        
    def write_pngs(self, image_dir):
        if len(self.pngs.items()) > 0:
            for k, v in self.pngs.items():
                f = open(op.join(image_dir,
                             k.replace('output', self.basename)),  'wb')
                f.write(v)
            f.close()
    
    def convert_code(self, to_convert):
        markdown = ''
        markdown += start_py_high
        markdown += add_newline(to_convert)
        return markdown 
    
    def convert_output(self, to_convert, prompt_number):
        markdown = ''
        if len(to_convert) == 0: 
            # Nothing to see here, move along
            return markdown
        for display_counter, output in enumerate(to_convert):
            if (output['output_type'] == 'pyout' or
                output['output_type'] == 'stream'):
                # I'm always going to assume this is a matplotlib output that I
                # want scrubbed out of existence (but still needs to be
                # counted!):
                if not output['text'].startswith('[<'):
                    # Add a tab in front, so that it gets rendered as 'tt':
                    markdown += '\t' + output['text']
            elif output['output_type'] == 'display_data':
                markdown += add_newline(
            '![png]({{ site.url }}/assets/images/%s/%s_%s_%s.png)\n'%(
                                                                self.basename,
                                                                self.basename,
                                                                prompt_number,
                                                                display_counter))
        return markdown
    
    def convert_text(self, txt):
        """ 
        This is here so that we can latexify math
        """
        new_txt = txt.replace('$', '$$')
        # And if we've backslashed it out, that means we want it there?
        new_text = new_txt.replace('\$$', '$')
        return new_text


    def convert(self):
        markdown = ''
        for idx, cell in enumerate(self.cells):
            # Per convention, the first cell is always the title of the post:
            if idx == 0:
                self.title = cell['source']
            else:
                if cell['cell_type'] == 'code':
                    this_str = add_newline(self.convert_code(cell['input']))
                    this_str += add_newline(self.convert_output(cell['outputs'],
                                                            idx))
                elif cell['cell_type'] == 'markdown':
                    this_str = add_newline(self.convert_text(cell['source'] ))
                markdown += this_str
        self.markdown = markdown
    
    def add_preamble(self, title, layout='post'):
        preamble = '---\nlayout: %s \ntitle:  "%s" \ndate:  %s \ncomments: true\n---\n'%(layout, title, self.basename[:10]) # Assume yyyy-mm-dd
        
        self.markdown = preamble + self.markdown
    
    def add_postscript(self, ps=None):
        """
        We'll want to add some text to all the notebooks, indicating where
        you'd download them
        """
        if ps is not None:
            self.markdown = add_newline(self.markdown) + add_newline(ps)

    def save_markdown(self, full_path):
        # Do it!
        md_file = open(full_path,'w')
        md_file.writelines(self.markdown)
        md_file.close()

if __name__ == "__main__":
    for f in os.listdir('./ipynb'):
        if f.endswith('.ipynb') and not f.startswith('draft'):
            print("processing %s"%f)
            basename = op.basename(f).split('.ipynb')[0]
            CMD = CustomMarkdownExporter()
            CMD.from_filename(op.join('ipynb', f))
            image_dir = op.join('content/images', basename)
            if not op.exists(image_dir):
                os.mkdir(image_dir)
            CMD.write_pngs(image_dir)
            CMD.convert()
            CMD.add_preamble(CMD.title)
            ps = '[Download this notebook]({{ site.url }}/ipynb/%s)'%(
                op.basename(f))
            CMD.add_postscript(ps)
            CMD.save_markdown('content/%s.md'%(basename))
