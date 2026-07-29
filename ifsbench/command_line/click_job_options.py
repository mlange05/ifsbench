# (C) Copyright 2020- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
click-based decorators for specifying job resource options.
"""

from enum import Enum
from functools import wraps
from typing import Optional

import click

from ifsbench.job import Job

__all__ = ['JobBuilder', 'job_options']


class JobBuilder(Job):
    """
    Helper class to build/override a :class:`Job`.

    This class is used to combine
        * job-related command line arguments
        * a default job (e.g. defined elsewhere in a script or config)
    in order to create a final job object. Only attributes that have been
    explicitly set on this builder (i.e. that are not ``None``) are used to
    override the corresponding attributes of the default job.
    """

    def build_job(self, default_job: Optional[Job] = None) -> Job:
        """
        Build a job, overriding attributes of a given default job.
        """

        job = default_job.clone() if default_job is not None else Job()

        overrides = self.model_dump(exclude_none=True)
        for key, value in overrides.items():
            setattr(job, key, value)

        return job


def job_options(func):
    """
    Decorator for use in click.commands to set job options.

    Adds a flag for every attribute of :class:`Job` (number of tasks/nodes,
    account, partition, CPU binding/distribution strategy, etc.).

    This decorator passes the different options as a `JobBuilder` object
    to the click.command-decorated function using the `job_builder`
    argument name.
    """

    @click.pass_context
    @wraps(func)
    def process_job_options(ctx, *args, **kwargs):
        # Only pass values that were actually set on the command line. Job
        # fields are typed as non-Optional with a `None` default, so passing
        # `None` explicitly would fail pydantic validation.
        values = {name: kwargs.pop(name) for name in Job.model_fields.keys()}
        builder = JobBuilder(**{k: v for k, v in values.items() if v is not None})

        return ctx.invoke(func, *args, **kwargs, job_builder=builder)

    # Build one click option per Job field, inferring the flag name and type
    # from the field itself so this stays in sync with Job automatically.
    # Options are added in reverse field order, since applying a decorator
    # in a loop is equivalent to stacking `@click.option`s bottom-up.
    for name, field_info in reversed(list(Job.model_fields.items())):
        option_name = '--' + name.replace('_', '-')
        annotation = field_info.annotation
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            option_type = click.Choice([member.value for member in annotation])
        else:
            option_type = annotation

        process_job_options = click.option(
            option_name,
            type=option_type,
            default=None,
            help=f'Override Job.{name}.',
        )(process_job_options)

    return process_job_options
