from streamlit.type_util import Key, to_key
from typing import cast, List, Optional, Union
from textwrap import dedent

import streamlit
from streamlit import config
from streamlit.logger import get_logger
from streamlit.proto.FileUploader_pb2 import FileUploader as FileUploaderProto
from streamlit.report_thread import get_report_ctx
from streamlit.state.widgets import register_widget, NoValue
from streamlit.state.session_state import (
    WidgetArgs,
    WidgetCallback,
    WidgetKwargs,
)
from .form import current_form_id
from ..proto.Common_pb2 import (
    FileUploaderState as FileUploaderStateProto,
    UploadedFileInfo as UploadedFileInfoProto,
)
from ..uploaded_file_manager import UploadedFile, UploadedFileRec
from .utils import check_callback_rules, check_session_state_rules

LOGGER = get_logger(__name__)

SomeUploadedFiles = Optional[Union[UploadedFile, List[UploadedFile]]]


class FileUploaderMixin:
    def file_uploader(
        self,
        label: str,
        type: Optional[Union[str, List[str]]] = None,
        accept_multiple_files: bool = False,
        key: Optional[Key] = None,
        help: Optional[str] = None,
        on_change: Optional[WidgetCallback] = None,
        args: Optional[WidgetArgs] = None,
        kwargs: Optional[WidgetKwargs] = None,
    ) -> SomeUploadedFiles:
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=None, key=key, writes_allowed=False)

        if type:
            if isinstance(type, str):
                type = [type]

            # May need a regex or a library to validate file types are valid
            # extensions.
            type = [
                file_type if file_type[0] == "." else f".{file_type}"
                for file_type in type
            ]

        file_uploader_proto = FileUploaderProto()
        file_uploader_proto.label = label
        file_uploader_proto.type[:] = type if type is not None else []
        file_uploader_proto.max_upload_size_mb = config.get_option(
            "server.maxUploadSize"
        )
        file_uploader_proto.multiple_files = accept_multiple_files
        file_uploader_proto.form_id = current_form_id(self.dg)
        if help is not None:
            file_uploader_proto.help = dedent(help)

        def deserialize_file_uploader(
            ui_value: Optional[FileUploaderStateProto], widget_id: str
        ) -> SomeUploadedFiles:
            file_recs = self._get_file_recs(widget_id, ui_value)
            if len(file_recs) == 0:
                return_value: Optional[Union[List[UploadedFile], UploadedFile]] = (
                    [] if accept_multiple_files else None
                )
            else:
                files = [UploadedFile(rec) for rec in file_recs]
                return_value = files if accept_multiple_files else files[0]
            return return_value

        def serialize_file_uploader(files: SomeUploadedFiles) -> FileUploaderStateProto:
            state_proto = FileUploaderStateProto()

            ctx = get_report_ctx()
            if ctx is None:
                return state_proto

            # ctx.uploaded_file_mgr._file_id_counter stores the id to use for
            # the *next* uploaded file, so the current highest file id is the
            # counter minus 1.
            state_proto.max_file_id = ctx.uploaded_file_mgr._file_id_counter - 1

            if not files:
                return state_proto
            elif not isinstance(files, list):
                files = [files]

            for f in files:
                file_info: UploadedFileInfoProto = state_proto.uploaded_file_info.add()
                file_info.id = f.id
                file_info.name = f.name
                file_info.size = f.size

            return state_proto

        # FileUploader's widget value is a list of file IDs
        # representing the current set of files that this uploader should
        # know about.
        widget_value, _ = register_widget(
            "file_uploader",
            file_uploader_proto,
            user_key=key,
            on_change_handler=on_change,
            args=args,
            kwargs=kwargs,
            deserializer=deserialize_file_uploader,
            serializer=serialize_file_uploader,
        )

        ctx = get_report_ctx()
        file_uploader_state = serialize_file_uploader(widget_value)
        uploaded_file_info = file_uploader_state.uploaded_file_info
        if ctx is not None and len(uploaded_file_info) != 0:
            newest_file_id = file_uploader_state.max_file_id
            active_file_ids = [f.id for f in uploaded_file_info]

            ctx.uploaded_file_mgr.remove_orphaned_files(
                session_id=ctx.session_id,
                widget_id=file_uploader_proto.id,
                newest_file_id=newest_file_id,
                active_file_ids=active_file_ids,
            )

        self.dg._enqueue("file_uploader", file_uploader_proto)
        return cast(SomeUploadedFiles, widget_value)

    @staticmethod
    def _get_file_recs(
        widget_id: str, widget_value: Optional[FileUploaderStateProto]
    ) -> List[UploadedFileRec]:
        if widget_value is None:
            return []

        ctx = get_report_ctx()
        if ctx is None:
            return []

        uploaded_file_info = widget_value.uploaded_file_info
        if len(uploaded_file_info) == 0:
            return []

        active_file_ids = [f.id for f in uploaded_file_info]

        # Grab the files that correspond to our active file IDs.
        return ctx.uploaded_file_mgr.get_files(
            session_id=ctx.session_id,
            widget_id=widget_id,
            file_ids=active_file_ids,
        )

    @property
    def dg(self) -> "streamlit.delta_generator.DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("streamlit.delta_generator.DeltaGenerator", self)
