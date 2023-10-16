
import os
import jinja2
import shlex
from gconst import SDS_ROOT, SLURM_TEMPLATE_ROOT, JOB_TEMP_DIR


class JobTemplateLoader:
    def __init__(self):
        tmpl_loader = jinja2.FileSystemLoader(searchpath=SLURM_TEMPLATE_ROOT)
        self._tmpl_env = jinja2.Environment(loader=tmpl_loader,
                                            autoescape=jinja2.select_autoescape(
                                                                disabled_extensions=('txt', 'moab', 'slurm', 'tmpl'),
                                                                default_for_string=True, default=True)
                                            )

    def create_job_file(self, template_name, filename, **kwargs):
        ''' Create a new job from a job template '''

        # shell-escape all string parameters
        tmpl_params = kwargs.copy()
        for key, value in kwargs.items():
            if isinstance(value, str):
                tmpl_params[key] = shlex.quote(value)
            elif isinstance(value, list):
                nl = []
                all_str = True
                for e in value:
                    if isinstance(e, str):
                        nl.append(shlex.quote(e))
                    else:
                        nl.append(e)
                        all_str = False
                tmpl_params[key] = nl
                if all_str:
                    tmpl_params['{}_STR'.format(key)] = ' '.join(nl)

        # render template
        tmpl = self._tmpl_env.get_template(template_name)
        result = tmpl.render(SDS_ROOT=shlex.quote(SDS_ROOT), **tmpl_params)

        # save job script
        os.makedirs(JOB_TEMP_DIR, exist_ok=True)
        job_fname = os.path.join(JOB_TEMP_DIR, filename)
        if os.path.exists(job_fname):
            os.remove(job_fname)
        with open(job_fname, 'w') as f:
            f.write(result)
            f.write('\n')
        return job_fname


def random_string(length=8):
    '''
    Generate a random alphanumerical string with length :length.
    '''
    import random
    import string

    return ''.join([random.choice(string.ascii_letters + string.digits) for n in range(length)])
